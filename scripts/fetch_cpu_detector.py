"""Download a pinned optional OpenCV Zoo CPU detector; no account or GPU needed."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT='48b10fa505aeb51fbd127fc60a5c660040e823b7'
BASE=f'https://raw.githubusercontent.com/opencv/opencv_zoo/{COMMIT}/models/object_detection_yolox/'
FILES={
 'model.onnx':(f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/object_detection_yolox/object_detection_yolox_2022nov.onnx','c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'),
 'yolox.py':(BASE+'yolox.py','c99d8f103d93d27d0fb1807c7b91cb32d2d6c4959326cb88c801f17d00468f84'),
 'LICENSE':(BASE+'LICENSE','0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088')}


def fetch(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);records={}
    for name,(url,expected) in FILES.items():
        p=root/name
        if p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(f'Existing {name} differs; refusing to overwrite')
        else:
            with urlopen(url,timeout=60) as response:data=response.read(40_000_000)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Download hash mismatch for {name}')
            p.write_bytes(data)
        records[name]={'url':url,'sha256':expected,'bytes':p.stat().st_size}
    (root/'manifest.json').write_text(json.dumps({'repo_commit':COMMIT,'files':records},indent=2))
    print(json.dumps({'status':'verified','output':str(root),'bytes':sum(r['bytes'] for r in records.values())}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/models/yolox');args=p.parse_args();fetch(args.output)
