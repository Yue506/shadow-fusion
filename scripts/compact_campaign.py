"""Same-image outputs and region-error diagnostics; full PNGs stay in local runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask


def font(size):
    for path in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Light.ttc',
                 '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def overlay(source,target,prediction=None):
    a=source.astype(float).copy()
    if prediction is None:
        a[target]=.35*a[target]+.65*np.array([30,205,190])
    else:
        for region,color in [(target&prediction,[30,205,190]),(prediction&~target,[245,140,30]),(target&~prediction,[205,40,160])]:
            a[region]=.25*a[region]+.75*np.array(color)
    return a.astype('uint8')


def compact(manifest,runs,output,ids=None,per_page=4,selection=None):
    records=load_manifest(manifest)
    if ids:
        byid={r['id']:r for r in records};records=[byid[i] for i in ids.split(',')]
    if any(r['split']=='final_test' for r in records) and not selection:
        raise ValueError('Final-test visualization requires the frozen selection file')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    width,cellheight=224,244;f=font(13)
    scoremaps={}
    for run in runs:
        p=Path(run).parent/'progress.jsonl'
        scoremaps[run]={r['id']:r for r in map(json.loads,p.read_text().splitlines()) if r['method']==Path(run).name}
    for start in range(0,len(records),per_page):
        chosen=records[start:start+per_page]
        canvas=Image.new('RGB',(width*(3+len(runs)),2*cellheight*len(chosen)),'#f4f4ef');d=ImageDraw.Draw(canvas)
        def cell(a,title,col,row):
            im=Image.fromarray(a) if isinstance(a,np.ndarray) else a
            im=ImageOps.contain(im.convert('RGB'),(218,212),Image.Resampling.LANCZOS)
            x,y=col*width,row*cellheight
            d.text((x+4,y+3),title,fill='#202329',font=f)
            canvas.paste(im,(x+(width-im.width)//2,y+28+(212-im.height)//2))
        for index,r in enumerate(chosen):
            source=read_rgb(r['source'])[0];g=read_rgb(r['generated'])[0];m=read_mask(r['mask']);target=read_mask(r['shadow_mask'])
            row=2*index
            cell(source,r['id']+' 原图',0,row);cell((m*255).astype('uint8'),'原物体 mask',1,row)
            cell(g,'冻结消除／去影候选',2,row)
            cell(overlay(source,target),'标注投影（青）',0,row+1)
            delta=np.abs(g.astype(float)-source).mean(-1)/40
            heat=np.zeros_like(source);heat[...,0]=(np.clip(delta,0,1)*255).astype('uint8')
            cell(heat,'候选与原图差值（40 封顶）',1,row+1)
            legend=Image.new('RGB',(218,212),'#f4f4ef');ld=ImageDraw.Draw(legend)
            for j,(label,color) in enumerate([('命中目标投影','#1ecdbE'),('误判背景','#f58c1e'),('漏检投影','#cd28a0')]):
                ld.rectangle((12,25+j*45,30,43+j*45),fill=color);ld.text((40,23+j*45),label,fill='black',font=f)
            cell(legend,'区域诊断（颜色见下）',2,row+1)
            for col,run in enumerate(runs,3):
                root=Path(run)/r['id'];s=scoremaps[run][r['id']]
                cell(Image.open(root/'output.png'),Path(run).name,col,row)
                pred=read_mask(root/'prediction.png')
                cell(overlay(source,target,pred),f'P{s["shadow_precision"]:.2f} R{s["shadow_recall"]:.2f} F1 {s["shadow_f1"]:.2f}',col,row+1)
        path=output/f'compact_{start//per_page+1:03d}.png';canvas.save(path);print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--ids');p.add_argument('--per-page',type=int,default=4)
    p.add_argument('--selection')
    a=p.parse_args();compact(a.manifest,a.runs,a.output,a.ids,a.per_page,a.selection)
