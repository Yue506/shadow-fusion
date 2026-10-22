"""Shadow-only residual segmentation. No labels, compositing or feathering here."""
from dataclasses import asdict, dataclass
import time
import numpy as np
from scipy import ndimage as ndi
from .core import LUMA
from .research import calibrate_srgb


@dataclass(frozen=True)
class ShadowMaskConfig:
    calibration: str = 'srgb_plane'
    absolute: float = 8.
    relative: float = .16
    weak_ratio: float = 1.
    link_radius: float = .04
    sigma: float = .65
    texture: bool = False

    def __post_init__(self):
        if self.calibration not in ('none', 'srgb_affine', 'srgb_plane'):
            raise ValueError('Unknown mask calibration')
        limits = {'absolute': (0,255), 'relative': (0,10), 'weak_ratio': (0,1),
                  'link_radius': (0,1), 'sigma': (0,8)}
        for key,(low,high) in limits.items():
            value=getattr(self,key)
            if type(value) not in (int,float) or not np.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{key} must be finite in [{low},{high}]')
        if self.absolute==0 or self.weak_ratio==0:
            raise ValueError('absolute and weak_ratio must be positive')
        if type(self.texture) is not bool:raise ValueError('texture must be boolean')


def mask_features(source, generated, object_mask, calibration='srgb_plane'):
    if (not isinstance(source,np.ndarray) or not isinstance(generated,np.ndarray)
        or source.dtype!=np.uint8 or generated.dtype!=np.uint8 or source.ndim!=3
        or source.shape[-1]!=3 or generated.shape!=source.shape):
        raise ValueError('Source and candidate must be matching uint8 RGB arrays')
    if (not isinstance(object_mask,np.ndarray) or object_mask.dtype!=bool
        or object_mask.shape!=source.shape[:2] or not object_mask.any()):
        raise ValueError('Object mask must be a nonempty matching boolean array')
    if calibration not in ('none','srgb_affine','srgb_plane'):
        raise ValueError('Unknown mask calibration')
    corrected,info=(generated,{'status':'disabled'}) if calibration=='none' else calibrate_srgb(
        source,generated,object_mask,spatial=calibration=='srgb_plane')
    delta=corrected.astype(float)-source
    brightness=delta@LUMA
    high=delta-ndi.gaussian_filter(delta,(1.5,1.5,0))
    return {'brightness':brightness, 'luma':source@LUMA,
            'texture':ndi.gaussian_filter(np.abs(high).mean(-1),.65),
            'distance':ndi.distance_transform_edt(~object_mask),
            'object_mask':object_mask, 'calibration':info}


def segment_features(features, config):
    """A weak component must contain strong evidence and approach the object."""
    m=features['object_mask'];short=min(m.shape)
    b=features['brightness']
    if config.sigma:b=ndi.gaussian_filter(b,config.sigma)
    relative=b/np.maximum(20,features['luma'])
    strong=(b>config.absolute)&(relative>config.relative)&~m
    weak=(b>config.absolute*config.weak_ratio)&(relative>config.relative*config.weak_ratio)&~m
    weak &= features['distance']<.65*short
    if config.texture:weak &= features['texture']<12+.3*np.maximum(b,0)
    labels,count=ndi.label(weak,np.ones((3,3)))
    indices=np.arange(1,count+1)
    keep=np.zeros(count+1,bool)
    if count:
        area=np.bincount(labels.ravel(),minlength=count+1)
        high=np.bincount(labels[strong].ravel(),minlength=count+1)
        nearest=ndi.minimum(features['distance'],labels,indices)
        keep[1:]=(area[1:]>=max(6,24*(short/512)**2)) & (high[1:]>=max(3,8*(short/512)**2)) & (nearest<=max(2,config.link_radius*short))
    prediction=keep[labels]&~m
    score=np.minimum(b/config.absolute,relative/max(1e-6,config.relative)) if config.relative else b/config.absolute
    score=np.clip(score,0,1).astype(np.float32);score[m]=0
    return prediction,score,{'candidate_components':count,'accepted_components':int(keep.sum())}


def predict_shadow(source, generated, object_mask, config=None):
    config=config if isinstance(config,ShadowMaskConfig) else ShadowMaskConfig(**(config or {}))
    start=time.perf_counter()
    features=mask_features(source,generated,object_mask,config.calibration)
    prediction,score,components=segment_features(features,config)
    return {'prediction':prediction,'score':score,'metadata':{
        'config':asdict(config),'calibration':features['calibration'],**components,
        'elapsed_cpu_seconds':time.perf_counter()-start,
        'semantics':'Target-associated positive residual; excludes object mask. Score is evidence, not calibrated probability.',
        'inputs':'Source, frozen removal candidate, object mask only; no ground truth.'}}
