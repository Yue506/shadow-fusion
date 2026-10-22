"""Development sweep and frozen-config shadow-only evaluation (labels never inference inputs)."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask,sha256
from shadow_fusion.research import compose_research
from shadow_fusion.shadow_mask import ShadowMaskConfig,mask_features,segment_features,predict_shadow


def metrics(p,t,m):
    assert not (p&m).any()
    tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
    bg=~(t|m)
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/max(1,tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                background_fpr=fp/max(1,int(bg.sum())),predicted_pixels=int(p.sum()),target_pixels=int(t.sum()))


def aggregate(rows):
    means={k:float(np.mean([r[k] for r in rows])) for k in ('iou','precision','recall','background_fpr')}
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    return {'count':len(rows),**means,'min_iou':min(r['iou'] for r in rows),
            'micro_iou':tp/max(1,tp+fp+fn),'micro_precision':tp/max(1,tp+fp),'micro_recall':tp/max(1,tp+fn),
            'tp':tp,'fp':fp,'fn':fn}


def load_data(manifests,cache):
    samples=[];provenance=[]
    cache.mkdir(parents=True,exist_ok=True)
    for manifest in manifests:
        provenance.append({'manifest_name':str(Path(manifest).name),'sha256':sha256(manifest)})
        for row in load_manifest(manifest):
            s=read_rgb(row['source'])[0];g=read_rgb(row['generated'])[0];m=read_mask(row['mask'])
            inputs={k:sha256(row[k]) for k in ('source','generated','mask')}
            key=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
            path=cache/f'{key}.npz'
            if path.exists():
                with np.load(path) as d:f={k:d[k] for k in d.files}
            else:
                f=mask_features(s,g,m)
                np.savez_compressed(path,**{k:v for k,v in f.items() if isinstance(v,np.ndarray)})
            # Labels loaded after inference features, never passed to mask_features/segment_features.
            target=read_mask(row['shadow_mask'])
            if (target&m).any():raise ValueError('Object/shadow labels overlap')
            samples.append({'row':row,'features':f,'target':target,'source':s,'generated':g,'mask':m,'hashes':inputs})
    return samples,provenance


def group_metrics(rows):
    return {cohort:aggregate([r for r in rows if r['cohort']==cohort]) for cohort in sorted({r['cohort'] for r in rows})}


def sweep(samples,output,grid=None):
    configs=[ShadowMaskConfig(**c) for c in grid] if grid is not None else [ShadowMaskConfig(absolute=a,relative=r,weak_ratio=w,link_radius=l)
             for a,r,w,l in itertools.product((6,10,16,24),(.08,.16,.24,.32),(1.,.5),(.02,.05))]
    scores=[]
    for index,c in enumerate(configs):
        rows=[]
        for d in samples:
            pred,_,_=segment_features(d['features'],c)
            rows.append({'id':d['row']['id'],'cohort':d['row'].get('cohort','actual'),**metrics(pred,d['target'],d['mask'])})
        groups=group_metrics(rows)
        # Equal cohort weight prevents 120 proxy inputs from drowning out 5 actual examples.
        score=float(np.mean([v['iou'] for v in groups.values()]))
        entry={'id':f'g{index:03d}','config':asdict(c),'objective_equal_cohort_macro_iou':score,'groups':groups,'rows':rows}
        scores.append(entry)
        if index%16==0:print(f'{index+1}/{len(configs)} objective={score:.4f}',flush=True)
    scores.sort(key=lambda r:r['objective_equal_cohort_macro_iou'],reverse=True)
    (output/'grid.json').write_text(json.dumps(scores,indent=2)+'\n')
    winner=scores[0]
    (output/'selected.json').write_text(json.dumps(winner['config'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in winner.items() if k!='rows'},indent=2),flush=True)


def evaluate(samples,methods,output,stability=False):
    summaries={};all_rows={}
    for name,entry in methods.items():
        rows=[]
        for n,d in enumerate(samples):
            row=d['row'];s=d['source'];g=d['generated'];m=d['mask']
            started=time.perf_counter()
            if entry['engine']=='mask':
                config=ShadowMaskConfig(**entry['config'])
                result=predict_shadow(s,g,m,config)
                p,score,meta=result['prediction'],result['score'],result['metadata']
            else:
                result=compose_research(s,g,m,entry['config']);p=(result['alpha']>.5)&~m;score=result['alpha'].copy();score[m]=0
                meta={'semantics':'Legacy thresholded compositing alpha > .5, object excluded'}
            stats={'id':row['id'],'scene_id':row['scene_id'],'cohort':row.get('cohort','actual'),
                   'source_name':row.get('name',row['id']),'groups':row.get('groups',[]),
                   'input_sha256':d['hashes'],**metrics(p,d['target'],m),'inference_cpu_seconds':time.perf_counter()-started}
            if stability and entry['engine']=='mask':
                rng=np.random.default_rng(20261023+n)
                perturbations={'brightness2':g.astype(float)+2,'color2':g.astype(float)+[2,-2,1],
                               'noise08':g.astype(float)+rng.normal(0,.8,g.shape)}
                for key,value in perturbations.items():
                    other=predict_shadow(s,np.rint(value).clip(0,255).astype('uint8'),m,config)['prediction']
                    stats['stability_'+key]=float((p&other).sum()/max(1,(p|other).sum()))
            dest=output/name/row.get('cohort','actual')/row['id'];dest.mkdir(parents=True)
            Image.fromarray(p.astype('uint8')*255).save(dest/'prediction.png')
            Image.fromarray(np.rint(score*255).clip(0,255).astype('uint8')).save(dest/'evidence.png')
            (dest/'metadata.json').write_text(json.dumps({'method':entry,'components':meta,'metrics':stats},indent=2)+'\n')
            rows.append(stats)
        summaries[name]=group_metrics(rows);all_rows[name]=rows
        (output/f'{name}_metrics.json').write_text(json.dumps({'method':entry,'groups':summaries[name],'rows':rows},indent=2)+'\n')
        print(name,json.dumps(summaries[name]),flush=True)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return all_rows


def main():
    started=time.time()
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache',default='outputs/shadow_mask_20261021/features');p.add_argument('--methods');p.add_argument('--sweep',action='store_true');p.add_argument('--grid');p.add_argument('--stability',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    assert Path(__import__('shadow_fusion').__file__).resolve().parents[2]==Path.cwd()
    data,provenance=load_data(a.manifest,Path(a.cache))
    (out/'provenance.json').write_text(json.dumps({'manifests':provenance,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'predictor_sha256':sha256('src/shadow_fusion/shadow_mask.py'),'runner_sha256':sha256(__file__),
        'selection':'Equal-weight cohort mean macro IoU; actual five are development regressions, proxies use label-derived candidates.',
        'count':len(data)},indent=2)+'\n')
    if a.sweep:sweep(data,out,json.loads(Path(a.grid).read_text()) if a.grid else None)
    if a.methods:evaluate(data,json.loads(Path(a.methods).read_text()),out,a.stability)
    (out/'execution.json').write_text(json.dumps({'argv':__import__('sys').argv,'start_unix':started,'end_unix':time.time(),
        'elapsed_seconds':time.time()-started,'device':'Apple M1 CPU','status':'complete'},indent=2)+'\n')

if __name__=='__main__':main()
