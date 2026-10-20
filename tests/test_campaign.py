import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scripts.prepare_desoba_campaign import prepare
from scripts.run_campaign_round import run


class CampaignTests(unittest.TestCase):
    def test_native_pair_preparation_split_and_locked_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/'dataset.zip'
            rng=np.random.default_rng(123)
            with ZipFile(archive,'w') as z:
                for i in range(5):
                    s=rng.integers(30,170,(128,128,3),dtype='uint8')
                    m=np.zeros((128,128),'uint8');m[40:60,40:60]=76
                    sh=np.zeros_like(m);sh[60:80,45:65]=76
                    d=s.copy();d[sh>0]=np.minimum(d[sh>0].astype(int)+30,255)
                    for folder,a in [('ShadowImage',s),('DeshadowedImage',d),('InstanceMask',m),('ShadowMask',sh)]:
                        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
                        z.writestr(f'DESOBA_DATASET/{folder}/scene{i}.png',b.getvalue())
            meta=prepare(archive,root/'prepared',count=5)
            self.assertEqual(meta['selected_count'],5)
            self.assertEqual(meta['split'],{'explore':3,'validation':1,'final_test':1})
            base=root/'prepared'
            rows=[json.loads(line) for line in (base/'target_proxy_all.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['scene_id'] for r in rows}),5)
            for row in rows:
                m=np.array(Image.open(base/row['mask']))>0
                sh=np.array(Image.open(base/row['shadow_mask']))>0
                self.assertFalse((m&sh).any())
                src=np.array(Image.open(base/row['source']))
                ref=np.array(Image.open(base/row['reference']))
                np.testing.assert_array_equal(src[~sh],ref[~sh])
            plan=root/'plan.json';plan.write_text('[{"id":"mask","config":{"mode":"mask"}}]')
            with self.assertRaisesRegex(ValueError,'Locked final test'):
                run(base/'target_proxy_final_test.jsonl',plan,root/'forbidden')
            self.assertFalse((root/'forbidden').exists())
            payload=run(base/'target_proxy_explore.jsonl',plan,root/'round',stability=True)
            self.assertEqual(payload['count'],3)
            self.assertEqual(payload['methods']['mask']['summary']['background_changed_fraction'],0)
            # Completed samples resume without appending duplicate records.
            run(base/'target_proxy_explore.jsonl',plan,root/'round',stability=True)
            self.assertEqual(len((root/'round/progress.jsonl').read_text().splitlines()),3)
            actual=json.loads((base/'target_proxy_explore.jsonl').read_text().splitlines()[0])
            actual['id']='r01_running'
            (base/'actual.jsonl').write_text(json.dumps(actual)+'\n')
            actual_result=run(base/'actual.jsonl',plan,root/'actual_round',stability=True)
            self.assertEqual(actual_result['count'],1)
            with self.assertRaises(FileExistsError): prepare(archive,base,count=5)


if __name__=='__main__':unittest.main()
