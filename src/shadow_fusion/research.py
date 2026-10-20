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
    calibration: str = 'linear'
    confidence: str = 'none'
    gate_low: float = 1.5
    gate_high: float = 4.

    def __post_init__(self):
        if self.feather not in ('none','single','multilevel','paper','adaptive','inward','inward_multilevel'):
            raise ValueError('Unknown feather strategy')
        make_config('shadow',color=self.color,extra=self.extra,gamma=self.gamma,
                    feather=self.feather if self.feather in ('none','single','multilevel','paper') else 'none')
        if self.association not in ('none','connected'):
            raise ValueError('Unknown association strategy')
        if type(self.link_radius) not in (int,float) or not np.isfinite(self.link_radius) or not 0 <= self.link_radius <= 1:
            raise ValueError('link_radius must be a finite fraction in [0,1]')
        if self.calibration not in ('linear','srgb_affine','srgb_plane'):
            raise ValueError('Unknown calibration strategy')
        if self.confidence not in ('none','positive'):
            raise ValueError('Unknown confidence strategy')
        if self.gamma and not self.extra:
            raise ValueError('Gamma requires extra-region detection')
        if any(type(v) not in (int,float) for v in (self.gate_low,self.gate_high)) or not np.isfinite(self.gate_low+self.gate_high) or not 0<=self.gate_low<self.gate_high<=255:
            raise ValueError('Confidence thresholds must satisfy 0 <= low < high <= 255')


def calibrate_srgb(source, generated, mask, spatial=False):
    """Robust same-coordinate RGB fit, optionally including a low-frequency plane."""
    yy,xx=np.indices(mask.shape)
    short=min(mask.shape)
    anchors=~tracefusion.dilate(mask,max(2,short*.04))
    hi_s=source.astype(float)-ndi.gaussian_filter(source.astype(float),(2,2,0))
    hi_g=generated.astype(float)-ndi.gaussian_filter(generated.astype(float),(2,2,0))
    anchors &= np.abs(hi_s-hi_g).mean(-1)<12
    holdout=anchors & ((xx+3*yy)%5==0)
    fit=anchors & ~holdout & (xx%2==0) & (yy%2==0)
    if fit.sum()<128:return generated.copy(),{'status':'insufficient_anchors','fit_pixels':int(fit.sum())}
    u=xx/max(1,mask.shape[1]-1)-.5;v=yy/max(1,mask.shape[0]-1)-.5
    g=generated.astype(float)/255;s=source.astype(float)/255
    designs=[np.column_stack((g[...,c][fit],np.ones(fit.sum()),u[fit],v[fit]))[:,:4 if spatial else 2] for c in range(3)]
    target=s[fit]
    prior=np.array([1.,0.,0.,0.])[:4 if spatial else 2]
    ridge=np.diag([2.,.02,1.,1.][:len(prior)])
    weights=np.ones(int(fit.sum()));params=np.tile(prior,(3,1))
    for _ in range(12):
        for c,design in enumerate(designs):
            params[c]=np.linalg.solve(design.T@(design*weights[:,None])+ridge,
                                      design.T@(weights*target[:,c])+ridge@prior)
            params[c,0]=np.clip(params[c,0],.75,1.33)
            params[c,1]=np.clip(params[c,1],-.15,.15)
            if spatial:params[c,2:]=np.clip(params[c,2:],-.08,.08)
        residual=np.column_stack([design@params[c]-target[:,c] for c,design in enumerate(designs)])
        deviation=np.sqrt(np.mean(residual**2,-1))
        sigma=max(.5/255,float(np.median(deviation))*1.4826)
        weights=np.minimum(1,1.5*sigma/np.maximum(deviation,1e-8))**2
    corrected=g*params[:,0]+params[:,1]
    if spatial:corrected+=u[...,None]*params[:,2]+v[...,None]*params[:,3]
    corrected=np.rint(corrected*255).clip(0,255).astype('uint8')
    return corrected,{'status':'fit','space':'sRGB','spatial_plane':spatial,'params':params.tolist(),
                      'fit_pixels':int(fit.sum()),'heldout_pixels':int(holdout.sum()),
                      'heldout_all_rgb_l1_before':float(np.abs(generated.astype(float)-source)[holdout].mean()) if holdout.any() else None,
                      'heldout_all_rgb_l1_after':float(np.abs(corrected.astype(float)-source)[holdout].mean()) if holdout.any() else None,
                      'heldout_scope':'All preselected anchors; includes unknown edits, not labelled background accuracy.'}


def render(source, corrected, support, feather, mask=None, confidence=None):
    short=min(support.shape)
    def confident(alpha):
        if confidence is not None:
            alpha=alpha*confidence;alpha[mask]=1
        return alpha
    def bound(envelope):
        return envelope & ((confidence>0)|mask) if confidence is not None else envelope
    if feather in ('inward','inward_multilevel'):
        if mask is None:raise ValueError('Inward feathering requires the original object mask')
        distance=ndi.distance_transform_edt(support)
        def inside_alpha(radius):
            t=np.clip((distance-.5)/max(1,radius),0,1)
            a=(t*t*(3-2*t)).astype(np.float32);a[mask]=1
            return confident(a)
        envelope=bound(support.copy())
        alpha=inside_alpha(short*(.04 if feather=='inward_multilevel' else .025))
        if feather=='inward_multilevel':
            s,g=core.linear(source),core.linear(corrected)
            s1,g1=(ndi.gaussian_filter(v,(max(1,short*.004),)*2+(0,)) for v in (s,g))
            s2,g2=(ndi.gaussian_filter(v,(max(2,short*.016),)*2+(0,)) for v in (s,g))
            out=np.zeros_like(s)
            for sb,gb,radius in [(s-s1,g-g1,short*.012),(s1-s2,g1-g2,short*.025),(s2,g2,short*.04)]:
                a=inside_alpha(radius)[...,None];out+=sb+a*(gb-sb)
            output=core.encode(out)
        else:output=tracefusion.blend(source,corrected,alpha)
        output[~envelope]=source[~envelope];output[mask]=corrected[mask]
        return output,alpha,envelope
    if feather=='none':
        alpha=support.astype(np.float32);envelope=support.copy()
    elif feather=='paper':
        alpha=tracefusion.feather(support);envelope=alpha>0
    elif feather=='adaptive':
        if mask is None:raise ValueError('Adaptive feathering requires the original object mask')
        scale=np.sqrt(mask.sum())
        envelope=tracefusion.dilate(support,max(1,min(short*.04,.12*scale)))
        alpha=core.feather(support,max(1,min(short*.025,.08*scale)),envelope)
    else:
        envelope=tracefusion.dilate(support,max(2,short*.04))
        alpha=core.feather(support,short*(.04 if feather=='multilevel' else .025),envelope)
    alpha=confident(alpha)
    envelope=bound(envelope)
    if feather=='multilevel' and confidence is not None:
        s,g=core.linear(source),core.linear(corrected)
        s1,g1=(ndi.gaussian_filter(v,(max(1,short*.004),)*2+(0,)) for v in (s,g))
        s2,g2=(ndi.gaussian_filter(v,(max(2,short*.016),)*2+(0,)) for v in (s,g))
        out=np.zeros_like(s)
        for sb,gb,radius in [(s-s1,g-g1,short*.012),(s1-s2,g1-g2,short*.025),(s2,g2,short*.04)]:
            a=confident(core.feather(support,radius,envelope))[...,None];out+=sb+a*(gb-sb)
        output=core.encode(out)
    elif feather=='multilevel':
        output=core.multilevel(source,corrected,support,envelope)
    else:output=tracefusion.blend(source,corrected,alpha)
    output[~envelope]=source[~envelope]
    if confidence is None:output[support]=corrected[support]
    else:output[mask]=corrected[mask]
    return output,alpha,envelope


def compose_research(source, generated, mask, settings=None):
    config=settings if isinstance(settings,ResearchConfig) else ResearchConfig(**(settings or {}))
    if (not isinstance(source,np.ndarray) or not isinstance(generated,np.ndarray)
            or source.dtype!=np.uint8 or generated.dtype!=np.uint8
            or source.ndim!=3 or source.shape[-1]!=3 or generated.shape!=source.shape):
        raise ValueError('Source and candidate must be matching uint8 RGB arrays')
    if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.shape!=source.shape[:2] or not mask.any():
        raise ValueError('Object mask must be a nonempty matching boolean array')
    started=time.perf_counter()
    custom_color=config.color and config.calibration!='linear'
    calibration=None
    if custom_color:
        generated,calibration=calibrate_srgb(source,generated,mask,config.calibration=='srgb_plane')
    result=compose(source,generated,mask,make_config('shadow',color=config.color and not custom_color,
                   extra=config.extra,gamma=config.gamma,
                   feather=config.feather if config.feather in ('none','single','multilevel','paper') else 'none'))
    result['metadata']['research_config']=asdict(config)
    result['metadata']['base_config']=result['metadata']['config']
    result['metadata']['config']={'mode':'shadow','engine':'research',**asdict(config)}
    if custom_color:result['metadata']['calibration']=calibration
    if (config.association=='none' or not config.extra) and config.feather in ('none','single','multilevel','paper') and config.confidence=='none':
        result['metadata']['elapsed_cpu_seconds']=time.perf_counter()-started
        return result
    short=min(mask.shape)
    center=tracefusion.dilate(mask,max(1,.008*short))
    support=result['support']
    if config.association=='connected' and config.extra:
        extra=support & ~center
        labels,count=ndi.label(extra,np.ones((3,3)))
        distance=ndi.distance_transform_edt(~mask)
        retained=np.zeros_like(mask)
        accepted=[]
        for k in range(1,count+1):
            region=labels==k
            if distance[region].min() <= max(2,config.link_radius*short):
                retained |= region;accepted.append(k)
        support=center|retained
        result['metadata']['association']={'candidate_components':count,'accepted_components':len(accepted)}
    gate=None
    if config.confidence=='positive':
        brightness=(result['corrected'].astype(float)-source)@core.LUMA
        t=np.clip((brightness-config.gate_low)/(config.gate_high-config.gate_low),0,1)
        gate=(t*t*(3-2*t)).astype(np.float32);gate[mask]=1
        result['metadata']['confidence']={'kind':'positive','low_255':config.gate_low,'high_255':config.gate_high,
                                          'nonzero_fraction':float((gate>0).mean())}
    output,alpha,envelope=render(source,result['corrected'],support,config.feather,mask,gate)
    if gate is not None:
        support=(support&(gate>=.5))|mask
        result['metadata']['support_semantics']='Geometric support restricted by confidence >= .5; blend alpha may be lower.'
    outside_count=int((~envelope).sum())
    outside_max=int(np.abs(output[~envelope].astype(np.int16)-source[~envelope].astype(np.int16)).max()) if outside_count else None
    assert outside_max in (0,None)
    result.update(output=output,alpha=alpha,envelope=envelope,support=support)
    result['metadata'].update(elapsed_cpu_seconds=time.perf_counter()-started,
                              edit_envelope_fraction=float(envelope.mean()),protected_pixel_count=outside_count,
                              outside_envelope_max_channel_difference=outside_max,
                              alpha_semantics=('Low-frequency alpha; edits bounded by support' if config.feather=='inward_multilevel'
                                               else 'Low-frequency alpha' if config.feather=='multilevel' else 'Blend alpha'))
    return result
