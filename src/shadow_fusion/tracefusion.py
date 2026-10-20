"""Paper reimplementation of LensErase TraceFusion, Appendix E.5; not official code."""
import numpy as np
from scipy import ndimage as ndi


def disk(radius):
    radius = max(1, round(radius))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def dilate(mask, radius):
    return ndi.binary_dilation(mask, structure=disk(radius))


def components(candidate, high, mask, scale):
    candidate = ndi.binary_opening(candidate, structure=disk(scale))
    labels, count = ndi.label(candidate, structure=np.ones((3, 3)))
    area = np.bincount(labels.ravel(), minlength=count + 1)
    strong = np.bincount(labels[high].ravel(), minlength=count + 1)
    overlap = np.bincount(labels[mask].ravel(), minlength=count + 1) > 0
    keep = ((area >= max(1, round(48 * scale**2))) &
            (strong >= max(1, round(8 * scale**2))) &
            (overlap | (strong >= area * .01)))
    keep[0] = False
    support = ndi.binary_closing(keep[labels], structure=disk(3 * scale))
    return dilate(support, 5 * scale)


def residual(source, generated, mask, scale):
    signed = ndi.gaussian_filter(generated.astype(np.float32) - source.astype(np.float32),
                                 (max(.5, 5 * scale), max(.5, 5 * scale), 0), mode="reflect")
    offset = np.median(signed[~mask], axis=0) if (~mask).any() else np.zeros(3)
    return np.max(np.abs(signed - offset), axis=-1)


def detect(source, generated, mask, gamma=True):
    h, w = mask.shape
    scale = min(h, w) / 512
    core = dilate(mask, 12 * scale)
    difference = residual(source, generated, core, scale)
    primary = core | components(difference > 4, difference > 10, core, scale)
    if not gamma:
        return core, primary & ~core, difference
    src_gamma = 255 * (source.astype(np.float32) / 255)**.75
    gen_gamma = 255 * (generated.astype(np.float32) / 255)**.75
    gamma_difference = residual(src_gamma, gen_gamma, core, scale)
    supplemental = components(gamma_difference > 4, gamma_difference > 10, core, scale) & ~primary
    labels, count = ndi.label(supplemental, structure=np.ones((3, 3)))
    area = np.bincount(labels.ravel(), minlength=count + 1)
    adjacent = np.bincount(labels[dilate(primary, scale)].ravel(), minlength=count + 1) > 0
    luma = np.array([.2126, .7152, .0722])
    dark = (source @ luma <= 96) & (generated @ luma <= 96)
    dark_area = np.bincount(labels[dark].ravel(), minlength=count + 1)
    keep = (area >= max(1, round(8 * scale**2))) & adjacent & (dark_area >= area * .5)
    keep[0] = False
    support = primary | keep[labels]
    return core, support & ~core, np.maximum(difference, gamma_difference)


def feather(support):
    if not support.any():
        return np.zeros(support.shape, np.float32)
    h, w = support.shape
    distance = ndi.distance_transform_edt(~support, sampling=(1 / h, 1 / w))
    return np.maximum(0, 1 - distance / .1).astype(np.float32)


def blend(source, generated, alpha):
    def linear(rgb):
        rgb = rgb.astype(np.float32) / 255
        return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    value = alpha[..., None] * linear(generated) + (1 - alpha[..., None]) * linear(source)
    srgb = np.where(value <= .0031308, 12.92 * value, 1.055 * np.maximum(value, 0)**(1 / 2.4) - .055)
    return np.rint(np.clip(srgb * 255, 0, 255)).astype(np.uint8)
