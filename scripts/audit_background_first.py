"""Post-inference colour, detail and true-boundary diagnostics on saved PNGs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask
from run_campaign_round import aggregate


def audit(manifest,run,output):
    run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError(output)
    records={r['id']:r for r in load_manifest(manifest)};rows=[]
    original=[json.loads(l) for l in (run/'progress.jsonl').read_text().splitlines()]
    for row in original:
        r=records[row['id']];root=run/row['method']/row['id']
        s=read_rgb(r['source'])[0].astype(float);o=read_rgb(root/'output.png')[0].astype(float)
        m=read_mask(r['mask']);t=read_mask(r['shadow_mask']);reference=read_rgb(r['reference'])[0].astype(float)
        bg=~(m|t);diff=np.abs(o-s)
        inner=t&~ndi.binary_erosion(t,iterations=2)
        outer=ndi.binary_dilation(t,iterations=2)&bg
        ideal=s.copy();ideal[t]=reference[t]
        band=(inner|outer)&~ndi.binary_dilation(m,iterations=1)
        high_s=s-ndi.gaussian_filter(s,(1,1,0));high_o=o-ndi.gaussian_filter(o,(1,1,0))
        gradient=np.mean([np.abs(ndi.sobel(o,axis=axis)/8-ndi.sobel(ideal,axis=axis)/8) for axis in [0,1]],axis=0)
        detail={'background_highpass_l1_255':float(np.abs(high_s-high_o)[bg].mean()),
                'shadow_inner_boundary_rgb_l1_255':float(np.abs(o-reference)[inner].mean()) if inner.any() else None,
                'protected_outer_boundary_rgb_l1_255':float(diff[outer].mean()) if outer.any() else None,
                'target_boundary_gradient_l1_255':float(gradient[band].mean()) if band.any() else None}
        e=read_mask(root/'envelope.png');assert np.array_equal(o[~e],s[~e])
        assert abs(float(np.any(diff[bg]!=0,-1).mean())-row['background_changed_fraction'])<1e-12
        rows.append({**row,**detail})
    payload={'protocol':'Post-inference only. Background highpass removes Gaussian sigma1; inner/outer target boundary is 2px; Sobel/8 gradient error against source with labelled target replaced by reference. Labels are evaluation-only.',
             'summary':{m:aggregate([r for r in rows if r['method']==m]) for m in sorted({r['method'] for r in rows})},'rows':rows}
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print('Audited',len(rows),'outputs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.manifest,a.run,a.output)
