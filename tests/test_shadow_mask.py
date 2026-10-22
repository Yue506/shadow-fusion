import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from shadow_fusion.shadow_mask import predict_shadow,ShadowMaskConfig
from shadow_fusion.cli import main


class ShadowMaskTests(unittest.TestCase):
    def fixture(self):
        s=np.full((96,96,3),100,'uint8');g=s.copy()
        m=np.zeros((96,96),bool);m[20:40,40:55]=True
        target=np.zeros_like(m);target[40:70,43:53]=True
        g[target]=150;g[m]=180
        g[70:90,75:90]=150
        return s,g,m,target

    def test_only_attached_shadow_no_object_halo_or_unrelated_change(self):
        s,g,m,t=self.fixture();before=s.copy()
        r=predict_shadow(s,g,m,dict(calibration='none',sigma=0))
        np.testing.assert_array_equal(r['prediction'],t)
        np.testing.assert_array_equal(s,before)
        self.assertFalse((r['prediction']&m).any())
        empty=predict_shadow(s,s,m,dict(calibration='none'))
        self.assertFalse(empty['prediction'].any())

    def test_hysteresis_keeps_weak_connected_shadow_not_isolated_noise(self):
        s,g,m,t=self.fixture();g[55:70,43:53]=110;g[5:12,5:12]=110
        config=dict(calibration='none',sigma=0,absolute=15,relative=.15,weak_ratio=.5)
        r=predict_shadow(s,g,m,config)
        np.testing.assert_array_equal(r['prediction'],t)

    def test_validation(self):
        for values in ({'sigma':float('nan')},{'relative':-.1},{'absolute':0},{'texture':1}):
            with self.assertRaises(ValueError):ShadowMaskConfig(**values)
        s,g,m,_=self.fixture()
        with self.assertRaises(ValueError):predict_shadow(s,g,m.astype('uint8'))
        with self.assertRaises(ValueError):predict_shadow(s,g[:-1],m)

    def test_cli_outputs_only_binary_mask_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);s,g,m,t=self.fixture()
            for name,a in [('s',s),('g',g),('m',m.astype('uint8')*255)]:Image.fromarray(a).save(p/f'{name}.png')
            (p/'config.json').write_text(json.dumps({'calibration':'none','sigma':0}))
            args=['predict-mask','--source',str(p/'s.png'),'--generated',str(p/'g.png'),'--mask',str(p/'m.png'),
                  '--output',str(p/'pred.png'),'--mask-config',str(p/'config.json')]
            self.assertEqual(main(args),0)
            np.testing.assert_array_equal(np.array(Image.open(p/'pred.png')),t.astype('uint8')*255)
            self.assertEqual(json.loads((p/'pred.json').read_text())['config']['sigma'],0)
            with self.assertRaises(SystemExit):main(args)

if __name__=='__main__':unittest.main()
