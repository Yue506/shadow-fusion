"""Optional CPU object proposals and competing shadow-caster seeds.

No dataset annotations enter this module. The OpenCV Zoo detector is downloaded
separately with its Apache-2.0 license; its reviewed wrapper is hash verified.
"""
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

MODEL_SHA='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
WRAPPER_SHA='c99d8f103d93d27d0fb1807c7b91cb32d2d6c4959326cb88c801f17d00468f84'
_models={}
_detections={}


def detect_objects(source, model_dir):
    import cv2
    root=Path(model_dir).resolve()
    key=(str(root),hashlib.sha256(source.tobytes()).hexdigest(),source.shape)
    if key in _detections:
        return _detections[key].copy(),{'cache_hit':True,'detection_seconds':0.,'model_sha256':MODEL_SHA}
    start=time.perf_counter()
    if str(root) not in _models:
        for name,expected in [('model.onnx',MODEL_SHA),('yolox.py',WRAPPER_SHA)]:
            path=root/name
            if not path.is_file():raise FileNotFoundError(f'Missing {path}; run scripts/fetch_cpu_detector.py')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'Unverified CPU detector file: {name}')
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(1)
        spec=importlib.util.spec_from_file_location('_shadow_fusion_yolox',root/'yolox.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _models[str(root)]=module.YoloX(str(root/'model.onnx'),confThreshold=.35,nmsThreshold=.5,
                                      backendId=cv2.dnn.DNN_BACKEND_OPENCV,targetId=cv2.dnn.DNN_TARGET_CPU)
    h,w=source.shape[:2];ratio=min(640/h,640/w)
    resized=cv2.resize(source,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
    padded=np.full((640,640,3),114,np.float32);padded[:resized.shape[0],:resized.shape[1]]=resized
    boxes=_models[str(root)].infer(padded)
    if len(boxes):boxes[:,:4]/=ratio
    else:boxes=np.empty((0,6),np.float32)
    _detections[key]=boxes.copy()
    return boxes,{'cache_hit':False,'detection_seconds':time.perf_counter()-start,
                  'model_sha256':MODEL_SHA,'backend':'OpenCV CPU; OpenCL disabled'}


def other_seeds(mask, boxes, evidence=None):
    """Match the user object and put competing seeds near other box bottoms."""
    h,w=mask.shape
    candidates=[]
    for index,box in enumerate(boxes):
        x,y,bw,bh=box[:4]
        x0,y0=max(0,int(x)),max(0,int(y));x1,y1=min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)))
        area=max(0,x1-x0)*max(0,y1-y0)
        if area<4:continue
        intersection=int(mask[y0:y1,x0:x1].sum())
        union=area+int(mask.sum())-intersection
        candidates.append({'index':index,'box':(x0,y0,x1,y1),'iou':intersection/union,
                           'coverage':intersection/mask.sum(),'inside_fraction':intersection/area})
    matched=None
    if candidates:
        best=max(candidates,key=lambda c:c['iou'])
        if best['iou']>=.1 or (best['iou']>=.05 and best['coverage']>=.7):matched=best['index']
    seeds=np.zeros_like(mask);used=[];seen=[]
    for c in candidates:
        if c['index']==matched or c['inside_fraction']>.5:continue
        seen.append(c['index'])
        x0,y0,x1,y1=c['box'];bw=x1-x0;bh=y1-y0
        xa,xb=int(x0+.15*bw),max(int(x0+.15*bw)+1,int(x1-.15*bw))
        ya=max(y0,int(y1-.15*bh));region=np.zeros_like(mask);region[ya:y1,xa:xb]=True
        if evidence is not None:
            contact=ndi.binary_dilation(region,iterations=max(2,round(min(h,w)*.025)))
            if (contact&(evidence>8)).sum()<max(3,round(8*(min(h,w)/512)**2)):continue
        seeds |= region;used.append(c['index'])
    seeds=ndi.binary_dilation(seeds,iterations=max(1,round(min(h,w)*.008))) & ~ndi.binary_dilation(mask,iterations=1)
    return seeds,{'detections':len(boxes),'other_boxes':len(seen),'active_other_boxes':len(used),'matched_target_index':matched,
                  'other_seed_pixels':int(seeds.sum())}


def grid_graph(cost):
    h,w=cost.shape;grid=np.arange(h*w,dtype=np.int32).reshape(h,w)
    rows,cols,weights=[],[],[]
    for a,b,factor in [(grid[:,:-1],grid[:,1:],1.),(grid[:-1],grid[1:],1.),
                       (grid[:-1,:-1],grid[1:,1:],np.sqrt(2)),(grid[:-1,1:],grid[1:,:-1],np.sqrt(2))]:
        aa=a.ravel();bb=b.ravel();value=(cost.ravel()[aa]+cost.ravel()[bb])*.5*factor
        rows.extend([aa,bb]);cols.extend([bb,aa]);weights.extend([value,value])
    return coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(h*w,h*w)).tocsr()


def competitive_support(mask, support, brightness, seeds, method='geodesic', margin=.9):
    if not seeds.any():return None
    if method=='euclidean':
        target_distance=ndi.distance_transform_edt(~mask)
        other_distance=ndi.distance_transform_edt(~seeds)
    elif method=='geodesic':
        evidence=ndi.gaussian_filter(brightness,max(.65,min(mask.shape)/512))
        graph=grid_graph((1+8/(1+np.maximum(evidence,0))).astype(np.float32))
        target_distance=dijkstra(graph,directed=True,indices=np.flatnonzero(mask),min_only=True).reshape(mask.shape)
        other_distance=dijkstra(graph,directed=True,indices=np.flatnonzero(seeds),min_only=True).reshape(mask.shape)
    else:raise ValueError('Unknown ownership distance')
    return mask|(support&(target_distance<=margin*other_distance))


def ownership_support(source,corrected,mask,support,model_dir,method='geodesic',margin=.9):
    boxes,info=detect_objects(source,model_dir)
    brightness=(corrected.astype(float)-source)@np.array([.2126,.7152,.0722])
    evidence=ndi.gaussian_filter(brightness,max(.65,min(mask.shape)/512))
    seeds,seed_info=other_seeds(mask,boxes,evidence);info.update(seed_info)
    result=competitive_support(mask,support,brightness,seeds,method,margin)
    info.update(method=method,margin=margin,applied=result is not None)
    return result,info
