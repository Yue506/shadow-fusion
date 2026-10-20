"""Resumable CPU rounds with inference/label separation and locked-test guard."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from shadow_fusion import compose, make_config
from shadow_fusion.benchmark import load_manifest, evaluate
from shadow_fusion.io import read_rgb, read_mask, sha256


def infer(source, generated, mask, setting):
    if setting.get('engine', 'baseline') == 'baseline':
        return compose(source, generated, mask, make_config(**setting.get('config', {})))
    from shadow_fusion.research import compose_research
    return compose_research(source, generated, mask, setting['config'])


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum()/union) if union else 1.


def aggregate(rows):
    keys = sorted({k for r in rows for k,v in r.items() if type(v) in (int,float)})
    result = {k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}
    for k in ['shadow_f1','shadow_iou','shadow_recall','shadow_precision','stability_iou_mean']:
        values=[r[k] for r in rows if r.get(k) is not None]
        if values:
            result[k+'_min']=float(min(values)); result[k+'_p10']=float(np.quantile(values,.1))
    for k in ['background_changed_fraction','background_rgb_l1_255']:
        values=[r[k] for r in rows if r.get(k) is not None]
        if values: result[k+'_max']=float(max(values))
    return result


def run(manifest, plan, output, stability=False, selection=None):
    records = load_manifest(manifest)
    if any(r.get('split') == 'final_test' for r in records):
        if selection is None or not Path(selection).is_file():
            raise ValueError('Locked final test requires a precommitted selection JSON')
        selected=json.loads(Path(selection).read_text())
        if not selected.get('frozen_before_test'):
            raise ValueError('Selection must explicitly freeze before final testing')
        if selected.get('plan_sha256') != sha256(plan):
            raise ValueError('Final test plan does not match frozen selection')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    configurations = json.loads(Path(plan).read_text())
    assert isinstance(configurations,list) and configurations
    assert len({c['id'] for c in configurations}) == len(configurations)
    identity={'manifest_sha256':sha256(manifest),'plan_sha256':sha256(plan),'stability':stability}
    identity_file=output/'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('Existing round identity differs; use a new directory')
    identity_file.write_text(json.dumps(identity,indent=2))
    start=time.time()
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','diff','HEAD'],text=True)
    (output/'code.diff').write_text(dirty)
    (output/'plan.json').write_text(Path(plan).read_text())
    progress=output/'progress.jsonl'
    completed={}
    if progress.exists():
        for line in progress.read_text().splitlines():
            row=json.loads(line); completed[(row['method'],row['id'])]=row
    for setting in configurations:
        for index, record in enumerate(records):
            key=(setting['id'],record['id'])
            if key in completed: continue
            for name in ('source','generated','mask'):
                expected=record.get('input_sha256',{}).get(Path(record[name]).name)
                if expected and sha256(record[name])!=expected:
                    raise ValueError(f'Frozen {name} changed for {record["id"]}')
            source=read_rgb(record['source'])[0]; generated=read_rgb(record['generated'])[0]
            mask=read_mask(record['mask'])
            # Only source, candidate, user mask and settings enter inference.
            result=infer(source,generated,mask,setting)
            prediction=(result['alpha']>.5) & ~mask
            support=result['support'] & ~mask
            instability=[]
            if stability:
                sample_seed=(int(record['id'][1:]) if re.fullmatch(r'd\d+',record['id'])
                             else int.from_bytes(hashlib.sha256(record['id'].encode()).digest()[:4],'little'))
                rng=np.random.default_rng(20261021+sample_seed)
                variants=[generated.astype(float)+2,generated.astype(float)+np.array([2,-2,1]),
                          generated.astype(float)+rng.normal(0,.8,generated.shape)]
                for perturbed in variants:
                    perturbed=np.rint(perturbed).clip(0,255).astype('uint8')
                    res=infer(source,perturbed,mask,setting)
                    instability.append(iou(prediction,(res['alpha']>.5)&~mask))
            # Ground-truth inputs are first opened after all inference for the sample.
            for name in ('shadow_mask','reference'):
                expected=record.get('input_sha256',{}).get(Path(record[name]).name)
                if expected and sha256(record[name])!=expected:
                    raise ValueError(f'Frozen evaluation {name} changed for {record["id"]}')
            target=read_mask(record['shadow_mask']); reference=read_rgb(record['reference'])[0]
            metrics=evaluate(source,mask,result,target,reference)
            tp,fp,fn=int((support&target).sum()),int((support&~target).sum()),int((~support&target).sum())
            metrics.update(detector_support_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                           detector_support_precision=tp/(tp+fp) if tp+fp else 0.,
                           detector_support_recall=tp/(tp+fn) if tp+fn else 1.)
            diff=np.abs(result['output'].astype(float)-source)
            bg=~(mask|target)
            metrics['background_changed_over3_fraction']=float((diff[bg].max(-1)>3).mean())
            for label in ['other_objects','other_shadows']:
                path=Path(record['source']).parent/(label+'.png')
                if path.exists():
                    region=read_mask(path)&bg
                    metrics[label+'_changed_fraction']=float((diff[region].max(-1)>0).mean()) if region.any() else None
            if stability:
                metrics.update(stability_iou_mean=float(np.mean(instability)),stability_iou_min=float(min(instability)))
            row={'method':setting['id'],'id':record['id'],'scene_id':record['scene_id'],
                 'split':record['split'],'cohort':record.get('cohort'), 'groups':record.get('groups',[]),**metrics}
            root=output/setting['id']/record['id']; root.mkdir(parents=True,exist_ok=True)
            for name,a in [('output.png',result['output']),('prediction.png',prediction.astype('uint8')*255),
                           ('support.png',result['support'].astype('uint8')*255),
                           ('envelope.png',result['envelope'].astype('uint8')*255)]:
                Image.fromarray(a).save(root/name)
            (root/'metadata.json').write_text(json.dumps(result['metadata'],indent=2))
            with progress.open('a') as stream: stream.write(json.dumps(row)+'\n')
            completed[key]=row
            if (index+1)%10==0:print(f'{setting["id"]}: {index+1}/{len(records)}',flush=True)
    rows=list(completed.values()); methods={}
    for setting in configurations:
        selected=[r for r in rows if r['method']==setting['id']]
        groups=sorted({g for r in selected for g in r['groups']})
        methods[setting['id']]={'count':len(selected),'summary':aggregate(selected),
                               'groups':{g:{'count':sum(g in r['groups'] for r in selected),
                                            **aggregate([r for r in selected if g in r['groups']])} for g in groups},
                               'worst_f1_cases':sorted(selected,key=lambda r:r['shadow_f1'])[:5]}
    payload={**identity,'code_revision':code_revision,'dirty_diff_sha256':hashlib.sha256(dirty.encode()).hexdigest(),
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
             'start_unix':start,'end_unix':time.time(),'wall_seconds':time.time()-start,
             'perturbations':['all channels +2','RGB +[2,-2,1]','Gaussian sigma .8, seed 20261021 + d-index; other ids use SHA256-derived integer'] if stability else [],
             'count':len(records),'methods':methods}
    (output/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
    for method,data in methods.items():
        s=data['summary']; print(method,{k:round(s[k],5) for k in ['shadow_precision','shadow_recall','shadow_f1','shadow_iou','background_changed_fraction','cpu_seconds']})
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True);p.add_argument('--stability',action='store_true');p.add_argument('--selection')
    a=p.parse_args();run(a.manifest,a.plan,a.output,a.stability,a.selection)
