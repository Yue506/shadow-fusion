"""Compare the packaged pipeline with the six frozen local experiment variants."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from shadow_fusion import compose, make_config
from shadow_fusion.io import read_mask, read_rgb

p = argparse.ArgumentParser()
p.add_argument('--legacy-root', required=True, type=Path)
p.add_argument('--output', type=Path)
args = p.parse_args()
root = args.legacy_root
records = json.loads((root/'controlled_cases.json').read_text())+json.loads((root/'real_cases.json').read_text())
checks = []
for record in records:
    name = record['id']; case = root/'cases'/name
    source, _ = read_rgb(case/'original.png'); generated, _ = read_rgb(case/'generated.png'); mask = read_mask(case/'mask.png')
    variants = [
        ('mask', make_config('mask'), case/'original_mask.png'),
        ('tracefusion', make_config('tracefusion'), case/'tracefusion.png'),
        ('shadow_single', make_config('shadow'), case/'corrected_limited.png'),
        ('shadow_multilevel', make_config('shadow', feather='multilevel'), case/'corrected_multilevel.png'),
        ('shadow_single_gamma', make_config('shadow', gamma=True), root/'gamma_ablation/outputs'/name/'limited_on.png'),
        ('shadow_multilevel_gamma', make_config('shadow', gamma=True, feather='multilevel'), root/'gamma_ablation/outputs'/name/'multilevel_on.png'),
    ]
    for label, config, path in variants:
        result = compose(source, generated, mask, config)
        expected = np.asarray(Image.open(path).convert('RGB'))
        if not np.array_equal(result['output'], expected):
            raise AssertionError(f'Frozen output mismatch: {name}/{label}')
        checks.append({'case': name, 'variant': label, 'pixels_exact': True})
payload = {'status': 'passed', 'cases': len(records), 'outputs_verified': len(checks), 'checks': checks}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps({k: v for k, v in payload.items() if k != 'checks'}, indent=2))
