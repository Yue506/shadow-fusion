"""Gamma-only supplemental support, matching the frozen ten-image ablation."""
import numpy as np
from scipy import ndimage as ndi
from . import core as fusion


def gamma_supplement(source, corrected, mask, primary):
    """Original TF dark/adjacency gates; retain the fixed primary detector's search distance."""
    short=min(mask.shape); scale=short/512
    core=fusion.legacy.dilate(mask,12*scale)
    source_gamma=255*(source.astype(np.float32)/255)**.75
    generated_gamma=255*(corrected.astype(np.float32)/255)**.75
    residual=fusion.legacy.residual(source_gamma,generated_gamma,core,scale)
    proposed=fusion.legacy.components(residual>4,residual>10,core,scale) & ~primary
    proposed &= ndi.distance_transform_edt(~mask)<.65*short
    labels,count=ndi.label(proposed,np.ones((3,3)))
    areas=np.bincount(labels.ravel(),minlength=count+1)
    adjacent=np.bincount(labels[fusion.legacy.dilate(primary,scale)].ravel(),minlength=count+1)>0
    dark=(source@fusion.LUMA<=96)&(corrected@fusion.LUMA<=96)
    dark_count=np.bincount(labels[dark].ravel(),minlength=count+1)
    keep=(areas>=max(1,round(8*scale**2))) & adjacent & (dark_count>=areas*.5)
    keep[0]=False
    extra=keep[labels]
    assert not (extra & primary).any()
    info={'proposed_components':int(count),'accepted_components':int(keep.sum()),
          'new_support_pixels':int(extra.sum()),'both_images_dark_pixels':int(dark.sum())}
    return extra,info
