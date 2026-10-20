"""Frozen-manifest evaluation. Labels are read only after compositing."""
from dataclasses import asdict
import csv
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .pipeline import compose
from .io import read_rgb, read_mask, sha256, write_result


def load_manifest(path):
    path = Path(path).resolve()
    records = []
    ids = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', row['id']):
            raise ValueError(f'Invalid sample id on line {line_number}')
        if row['id'] in ids:
            raise ValueError(f'Duplicate sample id: {row["id"]}')
        ids.add(row['id'])
        for key in ('source', 'generated', 'mask'):
            if key not in row or not isinstance(row[key], str):
                raise ValueError(f'Missing {key} for {row["id"]}')
        for key in ('source', 'generated', 'mask', 'shadow_mask', 'reference'):
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise ValueError(f'Missing {key} file for {row["id"]}')
        records.append(row)
    if not records:
        raise ValueError('Manifest has no samples')
    return records


def evaluate(source, object_mask, result, shadow_mask=None, reference=None):
    output = result['output']
    difference = np.abs(output.astype(np.int16)-source.astype(np.int16))
    metrics = {'changed_fraction': float(np.any(difference != 0, -1).mean()),
               'envelope_fraction': float(result['envelope'].mean()),
               'cpu_seconds': result['metadata']['elapsed_cpu_seconds'],
               'outside_envelope_max_channel_difference': result['metadata']['outside_envelope_max_channel_difference'],
               'protected_pixel_count': result['metadata']['protected_pixel_count']}
    if shadow_mask is not None:
        if shadow_mask.shape != object_mask.shape or (shadow_mask & object_mask).any():
            raise ValueError('Evaluation shadow mask must match and exclude the object mask')
        eligible = ~object_mask
        predicted = (result['alpha'] > .5) & eligible
        target = shadow_mask & eligible
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        background = eligible & ~target
        metrics.update(shadow_pixels=int(target.sum()), shadow_tp=tp, shadow_fp=fp, shadow_fn=fn,
                       shadow_precision=tp/(tp+fp) if tp+fp else (1. if not target.any() else 0.),
                       shadow_recall=tp/(tp+fn) if tp+fn else None,
                       shadow_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                       shadow_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                       background_changed_fraction=float(np.any(difference[background] != 0, -1).mean()) if background.any() else None,
                       background_rgb_l1_255=float(difference[background].mean()) if background.any() else None,
                       background_max_channel_difference=int(difference[background].max()) if background.any() else None)
        if reference is not None and target.any():
            if reference.shape != source.shape:
                raise ValueError('Reference shape does not match source')
            metrics['shadow_reference_rgb_l1_255'] = float(np.abs(output.astype(float)-reference)[target].mean())
    return metrics


def save_compact(records, runs, output_dir, per_page=10):
    """Columns are source, mask, then the matching output from each run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 256, 292
    titles = ['Original', 'Object mask']+[Path(run).name for run in runs]
    for start in range(0, len(records), per_page):
        selected = records[start:start+per_page]
        canvas = Image.new('RGB', (width*len(titles), height*len(selected)), '#f4f4f1')
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(selected):
            paths = [row['source'], row['mask']]+[str(Path(run)/row['id']/'output.png') for run in runs]
            for column, (path, title) in enumerate(zip(paths, titles)):
                with Image.open(path) as im:
                    preview = ImageOps.contain(im.convert('RGB'), (250, 250), Image.Resampling.LANCZOS)
                x, y = column*width, row_index*height
                draw.text((x+4, y+3), f'{row["id"]} | {title}', fill='#202329')
                canvas.paste(preview, (x+(width-preview.width)//2, y+30+(250-preview.height)//2))
        canvas.save(output_dir/f'compact_{start//per_page+1:03d}.png')


def run_batch(manifest, output_dir, config, ablations=(), *, composer=compose):
    records = load_manifest(manifest)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Batch output directory must be empty; use a new run directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in records:
        source, profile = read_rgb(row['source'])
        generated, _ = read_rgb(row['generated'])
        object_mask = read_mask(row['mask'])
        result = composer(source, generated, object_mask, config)
        result['metadata'].update(id=row['id'], ablations=list(ablations),
                                  input_sha256={key: sha256(row[key]) for key in ('source', 'generated', 'mask')})
        write_result(result, output_dir/row['id']/'output.png', icc_profile=profile,
                     artifacts_dir=output_dir/row['id']/'maps')
        # Evaluation data never enters compose().
        shadow = read_mask(row['shadow_mask']) if row.get('shadow_mask') else None
        reference = read_rgb(row['reference'])[0] if row.get('reference') else None
        score = {'id': row['id'], 'scene_id': row.get('scene_id', row['id']), 'split': row.get('split'),
                 **evaluate(source, object_mask, result, shadow, reference)}
        rows.append(score)
        with (output_dir/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(score)+'\n')
    numeric = sorted({key for row in rows for key, value in row.items() if type(value) in (float, int)})
    summary = {key: float(np.mean([row[key] for row in rows if row.get(key) is not None])) for key in numeric}
    payload = {'count': len(rows), 'config': asdict(config), 'ablations': list(ablations),
               'manifest_sha256': sha256(manifest), 'aggregation': 'Macro means over defined per-image values; count fields are also per-image means.',
               'summary': summary, 'rows': rows}
    (output_dir/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n')
    keys = sorted({key for row in rows for key in row})
    with (output_dir/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader(); writer.writerows(rows)
    save_compact(records, [output_dir], output_dir/'compact')
    return {'output_dir': str(output_dir.resolve()), 'count': len(rows), 'summary': summary, 'config': asdict(config)}
