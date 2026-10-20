import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_rgb(path):
    with Image.open(path) as im:
        if im.mode != 'RGB':
            raise ValueError(f'Expected 8-bit RGB image, got {im.mode}: {path}')
        return np.asarray(im).copy(), im.info.get('icc_profile')


def read_mask(path):
    with Image.open(path) as im:
        values = np.asarray(im)
    if values.ndim == 3:
        if values.shape[-1] != 3 or not np.array_equal(values[..., :3], np.repeat(values[..., :1], 3, -1)):
            raise ValueError('Mask RGB channels must agree')
        values = values[..., 0]
    if values.ndim != 2 or not np.isin(values, [0, 1, 255]).all():
        raise ValueError('Mask must be binary (0/1 or 0/255)')
    return values > 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_result(result, output, *, overwrite=False, icc_profile=None, artifacts_dir=None):
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Output must be lossless PNG')
    metadata = output.with_suffix('.json')
    if not overwrite and (output.exists() or metadata.exists()):
        raise FileExistsError(f'Output already exists: {output}; use --overwrite explicitly')
    artifacts = Path(artifacts_dir) if artifacts_dir else None
    if artifacts and artifacts.exists() and any(artifacts.iterdir()) and not overwrite:
        raise FileExistsError(f'Artifacts directory is not empty: {artifacts}')
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result['output']).save(output, icc_profile=icc_profile)
    with Image.open(output) as im:
        assert np.array_equal(np.asarray(im), result['output']), 'PNG round-trip changed pixels'
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        arrays = {k: result[k] for k in ('alpha', 'support', 'envelope')}
        if result['gamma_added'] is not None:
            arrays['gamma_added'] = result['gamma_added']
        np.savez_compressed(artifacts/'maps.npz', **arrays)
        for key in ('support', 'envelope', 'gamma_added'):
            if result[key] is not None:
                Image.fromarray(result[key].astype('uint8')*255).save(artifacts/(key+'.png'))
        Image.fromarray(np.rint(result['alpha']*255).astype('uint8')).save(artifacts/'alpha.png')
    result['metadata']['output_sha256'] = sha256(output)
    metadata.write_text(json.dumps(result['metadata'], indent=2, ensure_ascii=False)+'\n')
    return metadata
