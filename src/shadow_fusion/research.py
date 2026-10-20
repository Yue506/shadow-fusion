"""Exploration-only strategies; frozen public presets remain unchanged."""
from dataclasses import asdict, dataclass
import time
import numpy as np
from scipy import ndimage as ndi

from . import core, tracefusion
from .pipeline import compose, make_config


@dataclass(frozen=True)
class ResearchConfig:
    color: bool = True
    extra: bool = True
    gamma: bool = False
    feather: str = 'single'
    association: str = 'none'
    link_radius: float = .025

    def __post_init__(self):
        make_config('shadow',color=self.color,extra=self.extra,gamma=self.gamma,feather=self.feather)
        if self.association not in ('none','connected'):
            raise ValueError('Unknown association strategy')
        if not np.isfinite(self.link_radius) or not 0 <= self.link_radius <= 1:
            raise ValueError('link_radius must be a finite fraction in [0,1]')


def render(source, corrected, support, feather):
    short=min(support.shape)
    if feather=='none':
        alpha=support.astype(np.float32);envelope=support.copy()
    elif feather=='paper':
        alpha=tracefusion.feather(support);envelope=alpha>0
    else:
        envelope=tracefusion.dilate(support,max(2,short*.04))
        alpha=core.feather(support,short*(.04 if feather=='multilevel' else .025),envelope)
    if feather=='multilevel':
        output=core.multilevel(source,corrected,support,envelope)
    else:output=tracefusion.blend(source,corrected,alpha)
    output[~envelope]=source[~envelope]
    output[support]=corrected[support]
    return output,alpha,envelope


def compose_research(source, generated, mask, settings=None):
    config=ResearchConfig(**(settings or {}))
    started=time.perf_counter()
    result=compose(source,generated,mask,make_config('shadow',color=config.color,
                   extra=config.extra,gamma=config.gamma,feather=config.feather))
    result['metadata']['research_config']=asdict(config)
    if config.association=='none' or not config.extra:return result
    short=min(mask.shape)
    center=tracefusion.dilate(mask,max(1,.008*short))
    extra=result['support'] & ~center
    labels,count=ndi.label(extra,np.ones((3,3)))
    distance=ndi.distance_transform_edt(~mask)
    retained=np.zeros_like(mask)
    accepted=[]
    for k in range(1,count+1):
        region=labels==k
        if distance[region].min() <= max(2,config.link_radius*short):
            retained |= region;accepted.append(k)
    support=center|retained
    output,alpha,envelope=render(source,result['corrected'],support,config.feather)
    outside_count=int((~envelope).sum())
    outside_max=int(np.abs(output[~envelope].astype(np.int16)-source[~envelope].astype(np.int16)).max()) if outside_count else None
    assert outside_max in (0,None)
    result.update(output=output,alpha=alpha,envelope=envelope,support=support)
    result['metadata'].update(elapsed_cpu_seconds=time.perf_counter()-started,
                              edit_envelope_fraction=float(envelope.mean()),protected_pixel_count=outside_count,
                              outside_envelope_max_channel_difference=outside_max,
                              association={'candidate_components':count,'accepted_components':len(accepted)})
    return result
