"""Post-inference error attribution on the five known panel-derived cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from shadow_fusion.io import read_rgb,read_mask

PANELS={'r01_running':5,'r02_penguin':6,'r03_aircraft':9,'r04_field':12,'r05_hardware':14}


def analyze(legacy,campaign,output):
    legacy=Path(legacy);campaign=Path(campaign);rows=[]
    roots={m:campaign/'final_actual_regression'/m for m in ['tracefusion','shadow','selected']}
    roots['conservative']=campaign/'actual_conservative_confirm'/'conservative'
    for sample,panel in PANELS.items():
        folder=legacy/'cases'/sample;source=read_rgb(folder/'original.png')[0]
        mask=read_mask(folder/'mask.png');target=read_mask(folder/'shadow_eval.png')
        image=np.asarray(Image.open(legacy/'data_sources'/f'data_processing_Visualization_Examples_{panel}.png').convert('RGB'))
        assert image.shape==(256,1536,3) and np.array_equal(image[:,256:512],source)
        objects=image[:,1024:1280].mean(-1)>127;shadows=image[:,1280:1536].mean(-1)>127
        bg=~(mask|target)
        categories={'other_object':objects&bg,'other_shadow':shadows&~objects&bg,
                    'remaining_background':~(objects|shadows)&bg}
        assert sum(int(a.sum()) for a in categories.values())==bg.sum()
        for method,root in roots.items():
            output_rgb=read_rgb(root/sample/'output.png')[0];pred=read_mask(root/sample/'prediction.png')
            changed=np.any(output_rgb!=source,-1)&bg;fp=pred&~target
            counts={k:{'label_pixels':int(a.sum()),'changed_pixels':int((changed&a).sum()),
                       'predicted_fp_pixels':int((fp&a).sum())} for k,a in categories.items()}
            assert sum(c['changed_pixels'] for c in counts.values())==changed.sum()
            rows.append({'id':sample,'method':method,'categories':counts})
    summary={}
    for method in roots:
        summary[method]={k:{metric:sum(r['categories'][k][metric] for r in rows if r['method']==method)
                           for metric in ['label_pixels','changed_pixels','predicted_fp_pixels']} for k in ['other_object','other_shadow','remaining_background']}
    payload={'scope':'Known five actual candidates only; panel labels used exclusively after frozen inference.',
             'aggregation':'Micro pixel totals; categories are disjoint and sum to labelled background.',
             'summary':summary,'rows':rows}
    Path(output).write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.legacy_root,a.campaign,a.output)
