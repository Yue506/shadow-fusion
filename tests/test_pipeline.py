import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from shadow_fusion import compose, make_config, Config
from shadow_fusion import core
from shadow_fusion.gamma import gamma_supplement
from shadow_fusion.benchmark import load_manifest


def fixture():
    y, x = np.indices((128, 160))
    clean = np.stack([70+x*.45, 90+y*.45, 110+x*.25], -1).astype('uint8')
    mask = (x>25)&(x<45)&(y>30)&(y<70)
    shadow = (x>45)&(x<100)&(y>65)&(y<85)
    source = clean.copy(); source[mask] = [30, 60, 180]
    source[shadow] = core.encode(core.linear(clean)[shadow]*.5)
    generated = core.encode(core.linear(clean)*[1.04,.96,1.06]+[.005,-.003,.002])
    return source, generated, mask, shadow, clean


class PipelineTests(unittest.TestCase):
    def test_mask_mode_preserves_every_unselected_pixel(self):
        source, generated, mask, _, _ = fixture()
        result = compose(source, generated, mask, make_config('mask'))
        np.testing.assert_array_equal(result['output'][~mask], source[~mask])
        np.testing.assert_array_equal(result['output'][mask], generated[mask])

    def test_calibration_recovers_controlled_color_and_protects_outside(self):
        source, generated, mask, shadow, clean = fixture()
        originals = [a.copy() for a in (source, generated, mask)]
        for feather in ('single', 'multilevel'):
            result = compose(source, generated, mask, make_config('shadow', feather=feather))
            self.assertLess(np.abs(result['output'].astype(float)-clean)[shadow].mean(),
                            np.abs(generated.astype(float)-clean)[shadow].mean())
            np.testing.assert_array_equal(result['output'][~result['envelope']], source[~result['envelope']])
            np.testing.assert_array_equal(result['output'][result['support']], result['corrected'][result['support']])
        for array, original in zip((source, generated, mask), originals):
            np.testing.assert_array_equal(array, original)

    def test_ablation_switches_have_effective_metadata(self):
        config = make_config('shadow', gamma=True, ablate=('color', 'extra', 'feather'))
        self.assertFalse(config.color); self.assertFalse(config.extra); self.assertFalse(config.gamma)
        self.assertEqual(config.feather, 'none')
        source, generated, mask, shadow, _ = fixture()
        result = compose(source, generated, mask, config)
        np.testing.assert_array_equal(result['output'][shadow], source[shadow])
        self.assertEqual(result['metadata']['calibration']['status'], 'disabled')

    def test_gamma_darkness_and_adjacency_gates(self):
        mask = np.zeros((192, 256), bool); mask[70:110,50:80] = True
        primary = core.legacy.dilate(mask, 2)
        weak = np.zeros_like(mask); weak[90:110,80:145] = True
        source = np.full((*mask.shape,3),20,'uint8'); generated = source.copy()
        source[weak] = 1; generated[weak] = 8
        added, _ = gamma_supplement(source, generated, mask, primary)
        self.assertGreater((added & weak & ~primary).sum(), 500)
        same, _ = gamma_supplement(generated, generated, mask, primary)
        self.assertFalse(same.any())
        source[weak] = 160; generated[weak] = 200
        bright, _ = gamma_supplement(source, generated, mask, primary)
        self.assertFalse(bright.any())

    def test_validation_and_full_image_mask(self):
        source, generated, mask, _, _ = fixture()
        with self.assertRaises(ValueError): compose(source.astype(float), generated, mask)
        with self.assertRaises(ValueError): compose(source, generated, np.zeros_like(mask))
        with self.assertRaises(ValueError): Config(gamma=True, extra=False)
        with self.assertRaises(ValueError): make_config(ablate=('typo',))
        full = compose(source, generated, np.ones_like(mask), make_config('mask'))
        self.assertEqual(full['metadata']['protected_pixel_count'], 0)
        self.assertIsNone(full['metadata']['outside_envelope_max_channel_difference'])

    def test_cli_roundtrip_and_input_overwrite_guard(self):
        source, generated, mask, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for name, array in [('source',source), ('generated',generated), ('mask',mask.astype('uint8')*255)]:
                Image.fromarray(array).save(d/(name+'.png'))
            command = [sys.executable, '-m', 'shadow_fusion', 'compose', '--source',str(d/'source.png'),
                       '--generated',str(d/'generated.png'),'--mask',str(d/'mask.png'),'--mode','shadow',
                       '--ablate','color,gamma,extra,feather','--artifacts-dir',str(d/'maps'),'--output',str(d/'out.png')]
            subprocess.run(command, check=True, capture_output=True)
            meta = json.loads((d/'out.json').read_text())
            self.assertFalse(meta['config']['color']); self.assertFalse(meta['config']['extra'])
            self.assertEqual(meta['outside_envelope_max_channel_difference'], 0)
            self.assertTrue((d/'maps/maps.npz').is_file())
            repeat = subprocess.run(command, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            command[-1] = str(d/'source.png')
            before = (d/'source.png').read_bytes()
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode, 0)
            self.assertEqual(before, (d/'source.png').read_bytes())

    def test_manifest_rejects_duplicate_or_unsafe_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'manifest.jsonl'
            p.write_text(json.dumps({'id':'../escape','source':'s','generated':'g','mask':'m'})+'\n')
            with self.assertRaises(ValueError): load_manifest(p)


if __name__ == '__main__': unittest.main()
