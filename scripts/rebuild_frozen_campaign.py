"""Reconstruct exactly the frozen scenes without needing the legacy ten-image assets."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from prepare_desoba_campaign import digest,read,proxy_candidates


def rebuild(archive,selection,metadata,output):
    expected=json.loads(Path(metadata).read_text())
    if digest(archive)!=expected['archive_sha256']:raise ValueError('Archive hash differs from frozen dataset')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    records=[json.loads(line) for line in Path(selection).read_text().splitlines()]
    assert len(records)==expected['selected_count'] and len({r['scene_id'] for r in records})==len(records)
    output.mkdir(parents=True)
    with ZipFile(archive) as z:
        for row in records:
            root=output/row['id'];root.mkdir()
            def load(folder,mode,filter):
                return np.array(read(z,f'DESOBA_DATASET/{folder}/{row["name"]}',mode).resize(tuple(row['size']),filter))
            source=load('ShadowImage','RGB',Image.Resampling.LANCZOS)
            deshadowed=load('DeshadowedImage','RGB',Image.Resampling.LANCZOS)
            obj=load('InstanceMask','L',Image.Resampling.NEAREST)
            sh=load('ShadowMask','L',Image.Resampling.NEAREST)
            mask=obj==row['target_id'];shadow=(sh==row['target_id'])&~mask
            reference,candidates,info=proxy_candidates(source,deshadowed,shadow,row['construction']['seed'])
            assert info==row['construction']
            images={**{k+'.png':v for k,v in candidates.items()},'source.png':source,'mask.png':mask.astype('uint8')*255,
                    'shadow.png':shadow.astype('uint8')*255,'reference.png':reference,
                    'other_objects.png':((obj>0)&~mask).astype('uint8')*255,
                    'other_shadows.png':((sh>0)&~shadow&~mask).astype('uint8')*255}
            for name,a in images.items():
                Image.fromarray(a).save(root/name)
                if digest(root/name)!=row['input_sha256'][name]:
                    raise ValueError(f'{row["id"]}/{name}: hash mismatch; check NumPy/Pillow versions')
    for cohort in ['target_proxy','all_shadow_proxy']:
        for split in ['explore','validation','final_test','all']:
            rows=[{**r,'generated':f'{r["id"]}/{cohort}.png','cohort':cohort} for r in records if split=='all' or r['split']==split]
            (output/f'{cohort}_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'dataset.json').write_text(Path(metadata).read_text())
    print(json.dumps({'status':'verified','scenes':len(records),'pngs_checked':8*len(records),'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',default='docs/results/campaign_20261021/frozen_selection.jsonl')
    p.add_argument('--metadata',default='docs/results/campaign_20261021/dataset_metadata.json')
    a=p.parse_args();rebuild(a.archive,a.selection,a.metadata,a.output)
