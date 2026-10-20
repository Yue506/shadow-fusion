"""Native DESOBA scenes and explicitly labelled reference-derived proxy candidates.

Labels are used here for dataset construction/stratification, never in inference.
The candidates do not measure a removal model's generalization or object removal.
"""
import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage as ndi
from scipy.fft import dctn

URL = 'https://www.dropbox.com/scl/fi/h9uoif9upyxzf5nf5tz8r/DESOBA_DATASET.zip?rlkey=6qpmtbdjwnexj28vlzdaagtvm&dl=1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def phash(image):
    a = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), float)
    a = dctn(a, norm='ortho')[:8, :8].ravel()[1:]
    return a > np.median(a)


def read(z, name, mode):
    with Image.open(io.BytesIO(z.read(name))) as im:
        return im.convert(mode)


def proxy_candidates(source, reference_all, shadow, seed):
    reference=source.copy();reference[shadow]=reference_all[shadow]
    local_rng=np.random.default_rng(seed)
    gain=local_rng.uniform(.965,1.035,3);bias=local_rng.uniform(-7,7,3)
    yy,xx=np.indices(shadow.shape)
    plane=(xx/max(1,shadow.shape[1]-1)-.5)*local_rng.uniform(-5,5,3)[:,None,None]
    plane+=(yy/max(1,shadow.shape[0]-1)-.5)*local_rng.uniform(-5,5,3)[:,None,None]
    noise=local_rng.normal(0,.35,source.shape)
    candidates={cohort:np.rint(base*gain+bias+plane.transpose(1,2,0)+noise).clip(0,255).astype('uint8')
                for cohort,base in [('target_proxy',reference),('all_shadow_proxy',reference_all)]}
    info={'gain':gain.tolist(),'bias_255':bias.tolist(),'noise_sigma_255':.35,
          'plane_max_coefficient_255':5,'seed':seed}
    return reference,candidates,info


def prepare(archive, output, legacy=None, count=100, seed=42, max_side=384, geometric_dedup=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Dataset output already exists: {output}')
    rng = np.random.default_rng(seed)
    excluded_hashes = []
    if legacy:
        for p in Path(legacy).glob('cases/r*/original.png'):
            excluded_hashes.append(phash(Image.open(p)))
    with ZipFile(archive) as z:
        folders = {}
        for folder in ('ShadowImage', 'DeshadowedImage', 'InstanceMask', 'ShadowMask'):
            folders[folder] = {Path(n).name: n for n in z.namelist()
                               if folder.lower() in [p.lower() for p in Path(n).parts[:-1]]
                               and Path(n).suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
                               and '__MACOSX' not in n}
        names = sorted(set.intersection(*(set(v) for v in folders.values())))
        if not names:
            raise ValueError(f'No matching native images, folder counts: { {k:len(v) for k,v in folders.items()} }')
        rng.shuffle(names)
        candidates, seen_hashes, excluded, seen_scenes = [], [], Counter(), set()
        for name in names:
            scene_id = re.sub(r'-\d+$', '', Path(name).stem)
            if scene_id in seen_scenes:
                excluded['same_scene_name_variant'] += 1; continue
            s = read(z, folders['ShadowImage'][name], 'RGB')
            native_size = s.size
            if min(s.size) < 128:
                excluded['too_small'] += 1; continue
            hashed = phash(s)
            if any(np.count_nonzero(hashed != h) <= 6 for h in excluded_hashes + seen_hashes):
                excluded['near_duplicate_or_legacy'] += 1; continue
            size = tuple(max(1, round(v * min(1, max_side / max(s.size)))) for v in s.size)
            s = np.array(s.resize(size, Image.Resampling.LANCZOS))
            d = np.array(read(z, folders['DeshadowedImage'][name], 'RGB').resize(size, Image.Resampling.LANCZOS))
            obj = np.array(read(z, folders['InstanceMask'][name], 'L').resize(size, Image.Resampling.NEAREST))
            sh = np.array(read(z, folders['ShadowMask'][name], 'L').resize(size, Image.Resampling.NEAREST))
            ids = np.intersect1d(np.unique(obj), np.unique(sh))
            ids = [int(i) for i in ids if i not in (0, 255)
                   and (obj == i).sum() >= 64 and ((sh == i) & (obj != i)).sum() >= 32]
            if not ids:
                excluded['no_usable_target_pair'] += 1; continue
            target_id = int(rng.choice(ids))
            m, target = obj == target_id, (sh == target_id) & (obj != target_id)
            gray = s @ [.2126, .7152, .0722]
            brightening = (d.astype(float) - s) @ [.2126, .7152, .0722]
            extent = float(ndi.distance_transform_edt(~m)[target].max() / min(m.shape))
            texture = float(np.abs(gray - ndi.gaussian_filter(gray, 2))[target].mean())
            nearby_other = float(((sh > 0) & ~target & (ndi.distance_transform_edt(~target) < .06 * min(m.shape))).sum() / target.sum())
            features = {'shadow_luma': float(gray[target].mean()), 'shadow_extent': extent,
                        'texture': texture, 'weak_shadow_fraction': float((brightening[target] < 15).mean()),
                        'nearby_other_shadow_ratio': nearby_other, 'paired_objects': len(ids)}
            candidates.append({'name': name, 'scene_id':scene_id, 'target_id': target_id, 'native_size': native_size,
                               'size': size, 'features': features, 'phash': hashed,
                               'source_archive_member': folders['ShadowImage'][name]})
            seen_hashes.append(hashed)
            seen_scenes.add(scene_id)
            if len(candidates) % 100 == 0:
                print(f'Indexed {len(candidates)} independent usable scenes', flush=True)
        # Assign descriptive, overlapping difficulty proxies before method evaluation.
        if len(candidates) < count:
            raise ValueError(f'Only {len(candidates)} independent usable scenes; requested {count}, exclusions {excluded}')
        feature_names = ['shadow_luma', 'shadow_extent', 'texture', 'weak_shadow_fraction', 'nearby_other_shadow_ratio']
        thresholds = {k: np.quantile([c['features'][k] for c in candidates], [.35, .65]).tolist() for k in feature_names}
        for c in candidates:
            f = c['features']
            c['groups'] = [label for label, yes in [
                ('dark', f['shadow_luma'] <= thresholds['shadow_luma'][0]),
                ('long', f['shadow_extent'] >= thresholds['shadow_extent'][1]),
                ('textured', f['texture'] >= thresholds['texture'][1]),
                ('weak_or_soft', f['weak_shadow_fraction'] >= thresholds['weak_shadow_fraction'][1]),
                ('strong_or_hard', f['weak_shadow_fraction'] <= thresholds['weak_shadow_fraction'][0]),
                ('neighbor_shadow', f['nearby_other_shadow_ratio'] > 0),
                ('multi_object', f['paired_objects'] > 1)] if yes]
        # Balance overlapping groups without scoring any inference output.
        selected, coverage = [], Counter()
        pool = candidates.copy()
        rejected_related = []
        # Conservative manual source-only clusters, spotted before any method evaluation.
        related_clusters = [set(['web-shadow0210','web-shadow0211','web-shadow0212']),
                            set(['web-shadow0447','web-shadow0448','web-shadow0449','web-shadow0450']),
                            set(['lssd763','SBU-test04'])]
        features_cache = {}
        if geometric_dedup:
            import cv2
            cv2.setNumThreads(1)
            cv2.setRNGSeed(seed)
            sift = cv2.SIFT_create(nfeatures=400)
            matcher = cv2.BFMatcher()
            def image_features(im):
                kp, desc = sift.detectAndCompute(np.asarray(im.convert('L')), None)
                return np.float32([p.pt for p in kp]), desc, im.width*im.height
            legacy_features = []
            if legacy:
                legacy_features = [image_features(Image.open(p)) for p in Path(legacy).glob('cases/r*/original.png')]
            def features(c):
                if c['name'] not in features_cache:
                    im = read(z,folders['ShadowImage'][c['name']],'RGB').resize(c['size'],Image.Resampling.LANCZOS)
                    features_cache[c['name']] = image_features(im)
                return features_cache[c['name']]
            def related(a,b):
                x,dx,area_x=a; y,dy,area_y=b
                if dx is None or dy is None or len(dx)<8 or len(dy)<8:return False
                matches=matcher.knnMatch(dx,dy,k=2)
                good=[m for m,n in matches if m.distance < .75*n.distance]
                # Repeated texture can map many query points to one point and yield
                # a degenerate homography; require distinct points and image coverage.
                unique={}
                for m in sorted(good,key=lambda m:m.distance):unique.setdefault(m.trainIdx,m)
                good=list(unique.values())
                if len(good)<8:return False
                _,inliers=cv2.findHomography(np.array([x[m.queryIdx] for m in good]),
                                             np.array([y[m.trainIdx] for m in good]),cv2.RANSAC,4)
                if inliers is None or inliers.sum()<8 or inliers.mean()<.35:return False
                keep=inliers.ravel().astype(bool)
                px=np.array([x[m.queryIdx] for m in good])[keep]
                py=np.array([y[m.trainIdx] for m in good])[keep]
                return cv2.contourArea(cv2.convexHull(px))>.02*area_x and cv2.contourArea(cv2.convexHull(py))>.02*area_y
        while len(selected) < count:
            if not pool:raise ValueError(f'Only {len(selected)} independent scenes after geometric review')
            index = max(range(len(pool)), key=lambda i: sum(1/(1+coverage[g]) for g in pool[i]['groups']))
            c = pool.pop(index)
            match = next((s['name'] for s in selected if any(c['scene_id'] in group and s['scene_id'] in group for group in related_clusters)),None)
            if geometric_dedup and match is None:
                f=features(c)
                if any(related(f,lf) for lf in legacy_features):match='legacy_photo'
                else:match=next((s['name'] for s in selected if related(f,features(s))),None)
            if match is not None:
                rejected_related.append({'rejected':c['name'],'related_to':match});continue
            selected.append(c); coverage.update(c['groups'])
        # Twenty similar-difficulty five-scene blocks: 3 exploration, 1 validation, 1 final.
        selected.sort(key=lambda c: (len(c['groups']), c['features']['shadow_extent'], c['features']['texture']))
        for start in range(0, count, 5):
            block = selected[start:start+5]
            splits = np.array(['explore', 'explore', 'explore', 'validation', 'final_test'])
            rng.shuffle(splits)
            for c, split in zip(block, splits):
                c['split'] = str(split)
        output.mkdir(parents=True)
        rows = []
        for index, c in enumerate(selected):
            name, size, target_id = c['name'], c['size'], c['target_id']
            source = np.array(read(z, folders['ShadowImage'][name], 'RGB').resize(size, Image.Resampling.LANCZOS))
            reference_all = np.array(read(z, folders['DeshadowedImage'][name], 'RGB').resize(size, Image.Resampling.LANCZOS))
            obj = np.array(read(z, folders['InstanceMask'][name], 'L').resize(size, Image.Resampling.NEAREST))
            sh = np.array(read(z, folders['ShadowMask'][name], 'L').resize(size, Image.Resampling.NEAREST))
            mask, shadow = obj == target_id, (sh == target_id) & (obj != target_id)
            assert mask.any() and shadow.any() and not (mask & shadow).any()
            sample = f'd{index+1:03d}'
            root = output / sample; root.mkdir()
            # Deterministic nuisance cast, spatial drift and noise, identical for all strategies.
            reference,candidates,construction=proxy_candidates(source,reference_all,shadow,seed*10000+index)
            for cohort, generated in candidates.items():
                Image.fromarray(generated).save(root / f'{cohort}.png')
            for filename, a in [('source.png', source), ('mask.png', mask.astype('uint8')*255),
                                ('shadow.png', shadow.astype('uint8')*255), ('reference.png', reference),
                                ('other_objects.png', ((obj > 0) & ~mask).astype('uint8')*255),
                                ('other_shadows.png', ((sh > 0) & ~shadow & ~mask).astype('uint8')*255)]:
                Image.fromarray(a).save(root / filename)
            row = {k: v for k, v in c.items() if k != 'phash'}
            row.update(id=sample, source=f'{sample}/source.png',
                       mask=f'{sample}/mask.png', shadow_mask=f'{sample}/shadow.png', reference=f'{sample}/reference.png',
                       candidate_category='author_reference_derived_proxy',
                       construction=construction,
                       input_sha256={p.name:digest(p) for p in root.glob('*.png')})
            rows.append(row)
        for cohort in ['target_proxy', 'all_shadow_proxy']:
            for split in ['explore', 'validation', 'final_test', 'all']:
                records = [{**row, 'generated':f'{row["id"]}/{cohort}.png', 'cohort':cohort}
                           for row in rows if split == 'all' or row['split'] == split]
                (output/f'{cohort}_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        metadata = {'source_url':URL, 'archive_sha256':digest(archive), 'native_count_common':len(names),
                    'usable_independent_pool':len(candidates), 'selected_count':len(rows),
                    'split':dict(Counter(r['split'] for r in rows)), 'seed':seed, 'max_side':max_side,
                    'group_counts':dict(coverage), 'group_thresholds':thresholds, 'excluded':dict(excluded),
                    'groups_are_quantitative_proxies':'Softness and cross-material semantics need visual review; not human labels.',
                    'dedup':'Scene suffix groups, 63-bit DCT pHash <=6, manual source-only related clusters; optional SIFT ratio .75 / unique training points / homography 4 px / >=8 inliers and >=.35 inlier ratio / convex hull >2% of both images.',
                    'geometric_dedup':geometric_dedup, 'related_rejections':rejected_related,
                    'candidate_limitation':'Reference-derived proxy; target construction uses labels; no removal model or object removal evaluated.'}
        (output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # Source-only overview: no held-out predictions or results are exposed.
        canvas = Image.new('RGB',(1600,1800),'#eeeeea'); draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(rows):
            with Image.open(output/row['source']) as im:
                im=ImageOps.contain(im,(155,155))
                x,y=(i%10)*160,(i//10)*180
                canvas.paste(im,(x+(160-im.width)//2,y+20+(155-im.height)//2))
                draw.text((x+3,y+3),f'{row["id"]} {row["split"][:3]}',fill='black')
        canvas.save(output/'sources_100.png')
        return metadata


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True); p.add_argument('--output',required=True)
    p.add_argument('--legacy-root'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--geometric-dedup',action='store_true',help='Requires optional opencv-python-headless')
    args=p.parse_args()
    print(json.dumps(prepare(args.archive,args.output,args.legacy_root,seed=args.seed,geometric_dedup=args.geometric_dedup),indent=2))
