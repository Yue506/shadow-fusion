"""Post-selection stress diagnostic on validation only; never reselect parameters."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import shift
from shadow_fusion.benchmark import load_manifest,evaluate
from shadow_fusion.io import read_rgb,read_mask,sha256
from run_campaign_round import infer,iou,aggregate


def run(data,plan,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    settings=[s for s in json.loads(Path(plan).read_text()) if s['id'] in ('shadow','selected')]
    assert len(settings)==2
    rows=[];identities={}
    for cohort in ['target_proxy','all_shadow_proxy']:
        manifest=Path(data)/f'{cohort}_validation.jsonl';records=load_manifest(manifest)
        assert len(records)==20 and all(r['split']=='validation' for r in records)
        identities[cohort]=sha256(manifest)
        for record in records:
            source=read_rgb(record['source'])[0];g=read_rgb(record['generated'])[0];m=read_mask(record['mask'])
            for name in ['source','generated','mask']:
                assert sha256(record[name])==record['input_sha256'][Path(record[name]).name]
            rng=np.random.default_rng(20261021+int(record['id'][1:]));buffer=io.BytesIO()
            Image.fromarray(g).save(buffer,format='JPEG',quality=90,subsampling=2);buffer.seek(0)
            variants={'jpeg90':np.array(Image.open(buffer).convert('RGB')),
                      'gamma095':255*(g.astype(float)/255)**.95,
                      'noise2':g.astype(float)+rng.normal(0,2,g.shape),
                      'shift1x':shift(g.astype(float),(0,1,0),order=1,mode='nearest',prefilter=False)}
            for setting in settings:
                nominal=infer(source,g,m,setting);pred=(nominal['alpha']>.5)&~m
                results={}
                for variant,candidate in variants.items():
                    candidate=np.rint(candidate).clip(0,255).astype('uint8')
                    result=infer(source,candidate,m,setting)
                    assert np.array_equal(result['output'][~result['envelope']],source[~result['envelope']])
                    results[variant]=result
                # Evaluation labels are first read after all perturbation inference.
                target=read_mask(record['shadow_mask']);reference=read_rgb(record['reference'])[0]
                for variant,result in results.items():
                    row={'id':record['id'],'cohort':cohort,'method':setting['id'],'perturbation':variant,
                         'region_iou_to_nominal':iou(pred,(result['alpha']>.5)&~m),
                         **evaluate(source,m,result,target,reference)}
                    rows.append(row)
                    with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(cohort,record['id'],flush=True)
    summary={}
    for cohort in identities:
        summary[cohort]={}
        for method in ['shadow','selected']:
            summary[cohort][method]={v:aggregate([r for r in rows if r['cohort']==cohort and r['method']==method and r['perturbation']==v]) for v in variants}
    payload={'protocol':'Post-selection diagnostic, 20 validation scenes per condition; no retuning. JPEG quality90 subsampling2; sRGB exponent .95; Gaussian sigma2 seed20261021+d-index; candidate shifted +1px x with nearest border. Labels remain source-aligned.',
             'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'plan_sha256':sha256(plan),'manifest_sha256':identities,'summary':summary}
    assert len(rows)==320
    (output/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.data,a.plan,a.output)
