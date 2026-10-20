"""Independent module switches around the frozen, pixel-verified baseline."""
from dataclasses import asdict, dataclass
import time
import numpy as np
from . import core, tracefusion
from .gamma import gamma_supplement

PRESETS = {
    'mask': {'color': False, 'extra': False, 'gamma': False, 'feather': 'none'},
    'tracefusion': {'color': False, 'extra': True, 'gamma': True, 'feather': 'paper'},
    'shadow': {'color': True, 'extra': True, 'gamma': False, 'feather': 'single'},
}
MODULES = ('color', 'extra', 'gamma', 'feather')


@dataclass(frozen=True)
class Config:
    mode: str = 'shadow'
    color: bool = True
    extra: bool = True
    gamma: bool = False
    feather: str = 'single'

    def __post_init__(self):
        if self.mode not in PRESETS or self.feather not in ('none', 'single', 'multilevel', 'paper'):
            raise ValueError('Unknown mode or feather method')
        if any(type(getattr(self, key)) is not bool for key in ('color', 'extra', 'gamma')):
            raise ValueError('Module switches must be booleans')
        if self.mode == 'mask' and self.extra:
            raise ValueError('mask mode cannot enable extra-region detection')
        if self.gamma and not self.extra:
            raise ValueError('Gamma supplements extra-region detection; enable extra or disable gamma')


def make_config(mode='shadow', *, ablate=(), **overrides):
    if mode not in PRESETS:
        raise ValueError(f'Unknown mode: {mode}')
    unknown = set(overrides) - set(MODULES)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    ablate = tuple(ablate)
    if set(ablate) - set(MODULES):
        raise ValueError(f'Unknown ablation: {sorted(set(ablate)-set(MODULES))}')
    values = {'mode': mode, **PRESETS[mode], **{k: v for k, v in overrides.items() if v is not None}}
    for module in ablate:
        values[module] = 'none' if module == 'feather' else False
    if not values['extra']:
        values['gamma'] = False
    return Config(**values)


def compose(source, generated, mask, config=None):
    config = config or make_config()
    if not isinstance(config, Config):
        raise ValueError('config must be Config or make_config(...)')
    if (not isinstance(source, np.ndarray) or not isinstance(generated, np.ndarray)
            or source.dtype != np.uint8 or generated.dtype != np.uint8
            or source.ndim != 3 or source.shape[-1] != 3 or generated.shape != source.shape):
        raise ValueError('Source and candidate must be matching uint8 RGB arrays')
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != source.shape[:2] or not mask.any():
        raise ValueError('Object mask must be a nonempty boolean array matching the image')
    start = time.perf_counter()
    if config.color:
        corrected, calibration, _, _ = core.calibrate(source, generated, mask)
    else:
        corrected = generated.copy()
        calibration = {'status': 'disabled'}
    gamma_added = None
    gamma_info = {}
    if config.mode == 'mask':
        support = mask.copy()
    elif config.mode == 'tracefusion':
        if config.extra:
            center, extra, _ = tracefusion.detect(source, corrected, mask, gamma=config.gamma)
            support = center | extra
        else:
            support = tracefusion.dilate(mask, 12 * min(mask.shape) / 512)
    else:
        if config.extra:
            support, _, _, _ = core.shadow_support(source, corrected, mask)
        else:
            support = tracefusion.dilate(mask, max(1, .008 * min(mask.shape)))
        gamma_added = np.zeros_like(mask)
        if config.gamma:
            gamma_added, gamma_info = gamma_supplement(source, corrected, mask, support)
            support = support | gamma_added
    short = min(mask.shape)
    if config.feather == 'none':
        alpha = support.astype(np.float32)
        envelope = support.copy()
    elif config.feather == 'paper':
        alpha = tracefusion.feather(support)
        envelope = alpha > 0
    else:
        envelope = tracefusion.dilate(support, max(2, short * .04))
        alpha = core.feather(support, short * (.04 if config.feather == 'multilevel' else .025), envelope)
    if config.feather == 'multilevel':
        output = core.multilevel(source, corrected, support, envelope)
    else:
        output = tracefusion.blend(source, corrected, alpha)
    output[~envelope] = source[~envelope]
    output[support] = corrected[support]
    outside_count = int((~envelope).sum())
    outside_max = int(np.abs(output[~envelope].astype(np.int16)-source[~envelope].astype(np.int16)).max()) if outside_count else None
    assert outside_max in (None, 0)
    assert np.isfinite(alpha).all() and alpha.min() >= 0 and alpha.max() <= 1
    return {'output': output, 'alpha': alpha, 'support': support, 'envelope': envelope,
            'corrected': corrected, 'gamma_added': gamma_added,
            'metadata': {'config': asdict(config), 'calibration': calibration, 'gamma': gamma_info,
                         'elapsed_cpu_seconds': time.perf_counter()-start,
                         'edit_envelope_fraction': float(envelope.mean()),
                         'protected_pixel_count': outside_count,
                         'outside_envelope_max_channel_difference': outside_max,
                         'preservation_scope': 'Outside predicted envelope only; semantic background may be misdetected.',
                         'alpha_semantics': 'Low-frequency alpha' if config.feather == 'multilevel' else 'Blend alpha'}}
