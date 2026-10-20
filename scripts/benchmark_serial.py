"""Serial compositor-only latency on a fixed random scene subset; no GT read."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask,sha256
from run_campaign_round import infer


def benchmark(manifests,plan_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    plan=json.loads(Path(plan_path).read_text());samples=[]
    for manifest in manifests:
        records=load_manifest(manifest)
        indices=np.random.default_rng(42).choice(len(records),12,replace=False)
        for i in indices:
            r=records[int(i)]
            samples.append((r,read_rgb(r['source'])[0],read_rgb(r['generated'])[0],read_mask(r['mask'])))
    for setting in plan:infer(samples[0][1],samples[0][2],samples[0][3],setting)
    rows=[]
    for i,(r,s,g,m) in enumerate(samples):
        for offset in range(len(plan)):
            setting=plan[(i+offset)%len(plan)]
            start=time.perf_counter();result=infer(s,g,m,setting);elapsed=time.perf_counter()-start
            assert result['output'].shape==s.shape and result['metadata']['outside_envelope_max_channel_difference'] in (0,None)
            rows.append({'id':r['id'],'cohort':r['cohort'],'method':setting['id'],'seconds':elapsed})
    summary={}
    for setting in plan:
        values=np.array([r['seconds'] for r in rows if r['method']==setting['id']])*1000
        assert len(values)==24
        summary[setting['id']]={'n':24,'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'mean_ms':float(values.mean())}
    payload={'protocol':'12 fixed seed-42 exploration scenes per condition; one warmup; rotated method order; computation only; excludes image IO and generation',
             'hardware':'Apple M1 CPU','python':platform.python_version(),'plan_sha256':sha256(plan_path),
             'manifest_sha256':[sha256(m) for m in manifests],'methods':summary,'rows':rows}
    output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifests',nargs=2,required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();benchmark(a.manifests,a.plan,a.output)
