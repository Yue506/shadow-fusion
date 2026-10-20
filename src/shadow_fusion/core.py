"""Frozen image shadow-aware calibration, detection and blending baseline."""
import numpy as np
from scipy import ndimage as ndi
from . import tracefusion as legacy
LUMA = np.array([.2126, .7152, .0722])


def linear(a):
    a = np.asarray(a, np.float32) / 255
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def encode(a):
    a = np.clip(a, 0, 1)
    return np.rint(255 * np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)).clip(0, 255).astype(np.uint8)


def calibrate(source, generated, mask):
    """Robust same-coordinate fit on unmasked background, with held-out pixels."""
    h, w = mask.shape
    yy, xx = np.indices(mask.shape)
    anchors = ~legacy.dilate(mask, max(2, min(h, w) * .04))
    # Reject changed geometry; illumination differences are predominantly low frequency.
    hi_s = source.astype(float) - ndi.gaussian_filter(source.astype(float), (2, 2, 0))
    hi_g = generated.astype(float) - ndi.gaussian_filter(generated.astype(float), (2, 2, 0))
    anchors &= np.abs(hi_s - hi_g).mean(-1) < 12
    holdout = anchors & ((xx + 3 * yy) % 5 == 0)
    fit = anchors & ~holdout & (xx % 2 == 0) & (yy % 2 == 0)
    if fit.sum() < 128:
        return generated.copy(), {'status': 'insufficient_anchors', 'gain': [1.] * 3, 'bias': [0.] * 3}, anchors, holdout
    x, y = linear(generated)[fit], linear(source)[fit]
    params = np.array([[1., 0.]] * 3)
    weights = np.ones(len(x))
    for _ in range(10):
        for c in range(3):
            design = np.column_stack((x[:, c], np.ones(len(x))))
            lhs = design.T @ (design * weights[:, None]) + np.diag([2., .02])
            rhs = design.T @ (weights * y[:, c]) + np.array([2., 0.])
            params[c] = np.linalg.solve(lhs, rhs)
        params[:, 0] = np.clip(params[:, 0], .75, 1.33)
        params[:, 1] = np.clip(params[:, 1], -.15, .15)
        residual = x * params[:, 0] + params[:, 1] - y
        deviation = np.sqrt((residual ** 2).mean(-1))
        sigma = max(.002, np.median(deviation) * 1.4826)
        weights = np.minimum(1, 1.5 * sigma / np.maximum(deviation, 1e-8)) ** 2
    corrected = encode(linear(generated) * params[:, 0] + params[:, 1])
    # The robust-fit inliers exclude most removed shadows, which are positive residuals.
    error = np.abs(corrected.astype(float) - source).mean(-1)
    reliable = anchors & (error < max(4., np.percentile(error[anchors], 60)))
    info = {'status': 'fit', 'gain': params[:, 0].tolist(), 'bias': params[:, 1].tolist(),
            'fit_pixels': int(fit.sum()), 'reliable_pixels': int(reliable.sum()),
            'heldout_rgb_l1_before': float(np.abs(generated.astype(float) - source)[holdout & reliable].mean()) if (holdout & reliable).any() else None,
            'heldout_rgb_l1_after': float(np.abs(corrected.astype(float) - source)[holdout & reliable].mean()) if (holdout & reliable).any() else None}
    return corrected, info, reliable, holdout


def shadow_support(source, corrected, mask):
    short = min(mask.shape)
    sigma = max(.65, short / 512)
    delta = corrected.astype(float) - source.astype(float)
    brightness = ndi.gaussian_filter(delta @ LUMA, sigma)
    hi = delta - ndi.gaussian_filter(delta, (1.5 * sigma, 1.5 * sigma, 0))
    texture_error = ndi.gaussian_filter(np.abs(hi).mean(-1), sigma)
    core = legacy.dilate(mask, max(1, .008 * short))
    distance = ndi.distance_transform_edt(~mask)
    candidate = (brightness > 3) & (texture_error < 12 + .3 * np.maximum(brightness, 0)) & ~core
    candidate &= distance < .65 * short
    candidate = ndi.binary_closing(candidate, structure=legacy.disk(max(1, short / 512)))
    labels, count = ndi.label(candidate, np.ones((3, 3)))
    extra = np.zeros_like(mask)
    for k in range(1, count + 1):
        region = labels == k
        strong = region & (brightness > 8)
        if region.sum() >= max(6, 24 * (short / 512) ** 2) and strong.sum() >= max(3, 8 * (short / 512) ** 2):
            extra |= region
    extra = ndi.binary_fill_holes(ndi.binary_closing(extra, structure=legacy.disk(max(1, short / 512)))) & ~core
    support = core | extra
    envelope = legacy.dilate(support, max(2, short * .04))
    return support, extra, envelope, brightness


def feather(support, radius, envelope):
    distance = ndi.distance_transform_edt(~support)
    t = np.clip(1 - distance / max(1, radius), 0, 1)
    alpha = (t * t * (3 - 2 * t)).astype(np.float32)
    alpha[~envelope] = 0
    alpha[support] = 1
    return alpha


def multilevel(source, generated, support, envelope):
    short = min(support.shape)
    s, g = linear(source), linear(generated)
    s1, g1 = (ndi.gaussian_filter(v, (max(1, short * .004),) * 2 + (0,)) for v in (s, g))
    s2, g2 = (ndi.gaussian_filter(v, (max(2, short * .016),) * 2 + (0,)) for v in (s, g))
    out = np.zeros_like(s)
    for sb, gb, radius in [(s-s1, g-g1, short*.012), (s1-s2, g1-g2, short*.025), (s2, g2, short*.04)]:
        a = feather(support, radius, envelope)[..., None]
        out += sb + a * (gb-sb)
    out = encode(out)
    out[~envelope] = source[~envelope]
    out[support] = generated[support]
    return out
