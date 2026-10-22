"""Evidence report and lossless masks from frozen mask-only runs; no inference/tuning."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask,sha256
from explore_shadow_mask import aggregate
from compact_campaign import font,overlay

METHODS=('previous_alpha','background_alpha','shadow_mask')
NAMES={'previous_alpha':'前轮宽回贴区域','background_alpha':'严格回贴区域','shadow_mask':'纯影子预测'}
COLORS=('#a8a8a8','#7884b4','#484878')


def build(root,report):
    root=Path(root);report=Path(report);assets=report/'assets';exp=report/'experiments'
    final=root/'final_new20';validation=root/'validation'
    raw={m:json.loads((final/f'{m}_metrics.json').read_text()) for m in METHODS}
    actual={m:json.loads((validation/f'{"mask_raw" if m=="shadow_mask" else m}_metrics.json').read_text()) for m in METHODS}
    rows={m:raw[m]['rows'] for m in METHODS}
    realrows={m:[r for r in actual[m]['rows'] if r['cohort']=='existing_actual_model_candidate'] for m in METHODS}
    realsummary={m:aggregate(realrows[m]) for m in METHODS}
    sensitivity={m:{c:aggregate([r for r in rows[m] if r['cohort']==c and r['id'] not in ('d017','d020')]) for c in ('target_proxy','all_shadow_proxy')} for m in METHODS}
    (exp/'sensitivity18.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    for p in final.glob('*.json'):shutil.copy(p,exp/f'final_{p.name}')
    for p in ('selection_protocol.json','holdout_protocol.json','actual5_timing.json','final_methods.json'):
        shutil.copy(root/p,exp/p)
    for p in ('selection.json','environment.json','heldout_selection.jsonl','heldout_dataset.json'):
        shutil.copy(Path('docs/results/shadow_mask_20261021')/p,exp/p)
    metrics_table='| 条件 | 方法 | n | IoU ↑ | Precision ↑ | Recall ↑ | 背景误检率 ↓ | 微平均IoU ↑ | 最差IoU ↑ |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for label,group in [('已知真实生成5图',realsummary),('新20图：仅目标去影代理',{m:raw[m]['groups']['target_proxy'] for m in METHODS}),('新20图：全图去影代理',{m:raw[m]['groups']['all_shadow_proxy'] for m in METHODS})]:
        for m in METHODS:
            s=group[m];metrics_table+=f'| {label} | {NAMES[m]} | {s["count"]} | {s["iou"]:.5f} | {s["precision"]:.5f} | {s["recall"]:.5f} | {100*s["background_fpr"]:.5f}% | {s["micro_iou"]:.5f} | {s["min_iou"]:.5f} |\n'
    summary=raw['shadow_mask']['groups'];timings={m:float(np.mean([r['inference_cpu_seconds'] for r in rows[m]]))*1000 for m in METHODS}
    stability={c:{k:float(np.mean([r[k] for r in rows['shadow_mask'] if r['cohort']==c])) for k in ('stability_brightness2','stability_color2','stability_noise08')} for c in summary}
    (exp/'stability_and_timing.json').write_text(json.dumps({'stability':stability,'mean_full_inference_ms':timings,'empty_prediction_count':{c:sum(r['predicted_pixels']==0 for r in rows['shadow_mask'] if r['cohort']==c) for c in summary},'nonempty_stability':{c:{k:float(np.mean([r[k] for r in rows['shadow_mask'] if r['cohort']==c and r['predicted_pixels']>0])) for k in stability[c]} for c in summary}},indent=2)+'\n')
    # Same-frame plates include every mask and overlay, never RGB compositing output.
    compact_paths=[]
    def plate(records,run,co,stem):
        out=[];width,height=210,238
        for start in range(0,len(records),4):
            chosen=records[start:start+4];im=Image.new('RGB',(width*6,height*2*len(chosen)),'#f4f4ef');d=ImageDraw.Draw(im);f=font(13)
            def cell(a,title,x,y):
                a=Image.fromarray(a) if isinstance(a,np.ndarray) else a
                preview=ImageOps.contain(a.convert('RGB'),(204,204),Image.Resampling.LANCZOS)
                d.text((x*width+4,y*height+3),title,fill='black',font=f)
                im.paste(preview,(x*width+(width-preview.width)//2,y*height+29+(204-preview.height)//2))
            for i,r in enumerate(chosen):
                s=read_rgb(r['source'])[0];g=read_rgb(r['generated'])[0];obj=read_mask(r['mask']);t=read_mask(r['shadow_mask']);y=2*i
                cell(s,r['id']+' 原图',0,y);cell(obj.astype('uint8')*255,'物体mask（排除）',1,y);cell(g,'冻结候选',2,y)
                cell(t.astype('uint8')*255,'目标影子真值',0,y+1)
                cell(overlay(s,t),'真值位置（青）',1,y+1)
                legend=Image.new('RGB',(204,204),'#f4f4ef');ld=ImageDraw.Draw(legend)
                for j,(label,color) in enumerate([('命中','#1ecdbe'),('误检','#f58c1e'),('漏检','#cd28a0')]):
                    ld.rectangle((8,30+j*42,24,46+j*42),fill=color);ld.text((32,28+j*42),label,fill='black',font=f)
                cell(legend,'误差图图例',2,y+1)
                for x,m in enumerate(METHODS,3):
                    name='mask_raw' if co=='existing_actual_model_candidate' and m=='shadow_mask' else m
                    p=read_mask(run/name/co/r['id']/'prediction.png')
                    stats=next(v for v in (realrows[m] if co=='existing_actual_model_candidate' else rows[m]) if v['id']==r['id'] and v['cohort']==co)
                    cell(p.astype('uint8')*255,NAMES[m]+f' IoU {stats["iou"]:.3f}',x,y)
                    cell(overlay(s,t,p),'同图误检／漏检诊断',x,y+1)
            path=assets/f'{stem}_{start//4+1:02d}.png';im.save(path);out.append(path)
        return out
    actual_records=load_manifest('data/regression_actual5.jsonl')
    compact_paths+=plate(actual_records,validation,'existing_actual_model_candidate','actual')
    for co in ('target_proxy','all_shadow_proxy'):
        records=load_manifest(f'data/shadow_mask_holdout20/{co}_heldout.jsonl')
        compact_paths+=plate(records,final,co,co)
    # Stable 3-cell per-method boards: genuine success, genuine failure, final mask.
    lookup={r['id']:r for r in load_manifest('data/shadow_mask_holdout20/all_shadow_proxy_heldout.jsonl')}
    for m in METHODS:
        rr=[r for r in rows[m] if r['cohort']=='all_shadow_proxy'];best=max(rr,key=lambda r:r['iou']);worst=min(rr,key=lambda r:r['iou'])
        im=Image.new('RGB',(1008,340),'#f4f4ef');d=ImageDraw.Draw(im);f=font(15)
        for x,entry in enumerate((best,worst,best)):
            r=lookup[entry['id']];s=read_rgb(r['source'])[0];t=read_mask(r['shadow_mask']);p=read_mask(final/m/'all_shadow_proxy'/r['id']/'prediction.png')
            a=p.astype('uint8')*255 if x==2 else overlay(s,t,p)
            preview=ImageOps.contain(Image.fromarray(a).convert('RGB'),(326,264));im.paste(preview,(x*336+(336-preview.width)//2,32+(264-preview.height)//2))
            d.text((x*336+5,4),NAMES[m]+'｜'+('相对成功','失败','输出mask')[x],fill='black',font=f)
            d.text((x*336+5,305),f'{r["id"]} IoU {entry["iou"]:.3f} P {entry["precision"]:.3f} R {entry["recall"]:.3f}',fill='black',font=f)
        im.save(assets/f'board_{m}.png')
    # Quantitative plot uses exact tables as source; latency includes preprocessing.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    path='/System/Library/Fonts/STHeiti Light.ttc';font_manager.fontManager.addfont(path);plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name();plt.rcParams['svg.fonttype']='none'
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for k,(title,data) in enumerate([('平均IoU（↑）',[realsummary,{m:raw[m]['groups']['target_proxy'] for m in METHODS},{m:raw[m]['groups']['all_shadow_proxy'] for m in METHODS}]),('新20图背景误检率（% ↓）',[{m:raw[m]['groups']['target_proxy'] for m in METHODS},{m:raw[m]['groups']['all_shadow_proxy'] for m in METHODS}])]):
        ax=axs[k]
        for i,m in enumerate(METHODS):
            vals=[g[m]['iou'] if k==0 else g[m]['background_fpr']*100 for g in data]
            ax.bar(np.arange(len(data))+(i-1)*.24,vals,.23,color=COLORS[i],label=NAMES[m])
        ax.set_xticks(range(len(data)),['已知真实5图','目标代理20','全图代理20'] if k==0 else ['目标代理20','全图代理20']);ax.set_title(title)
        if k==0:ax.set_ylim(0,1);ax.legend(fontsize=9)
    axs[2].bar(range(3),[timings[m] for m in METHODS],color=COLORS);axs[2].set_xticks(range(3),[NAMES[m] for m in METHODS],rotation=15);axs[2].set_title('完整CPU推理毫秒（↓）')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(assets/'metrics.png',dpi=160);fig.savefig(assets/'metrics.svg');plt.close(fig)
    ablation='| 开发消融 | 真实5图IoU | 全图代理60 IoU | 目标代理60 IoU |\n|---|---:|---:|---:|\n'
    for m,label in [('full','完整纯mask'),('no_calibration','关背景补偿'),('no_relative','关相对亮度'),('no_association','关目标关联'),('hysteresis','弱阈值扩展')]:
        a=json.loads((root/'R4_ablations'/f'{m}_metrics.json').read_text())['groups'];ablation+=f'| {label} | {a["existing_actual_model_candidate"]["iou"]:.5f} | {a["all_shadow_proxy"]["iou"]:.5f} | {a["target_proxy"]["iou"]:.5f} |\n'
    per_image='| 已知真实案例 | 旧宽区域IoU | 严格区域IoU | 纯mask IoU | 纯mask P | 纯mask R |\n|---|---:|---:|---:|---:|---:|\n'
    for r in realrows['shadow_mask']:
        old=next(x for x in realrows['previous_alpha'] if x['id']==r['id']);strict=next(x for x in realrows['background_alpha'] if x['id']==r['id'])
        per_image+=f'| {r["id"]} | {old["iou"]:.5f} | {strict["iou"]:.5f} | {r["iou"]:.5f} | {r["precision"]:.5f} | {r["recall"]:.5f} |\n'
    group_table='| 新20图全图代理分组 | n | IoU | Precision | Recall |\n|---|---:|---:|---:|---:|\n'
    for g in sorted({g for r in rows['shadow_mask'] for g in r['groups']}):
        selected=[r for r in rows['shadow_mask'] if r['cohort']=='all_shadow_proxy' and g in r['groups']]
        if not selected:continue
        a=aggregate(selected);group_table+=f'| {g} | {len(selected)} | {a["iou"]:.5f} | {a["precision"]:.5f} | {a["recall"]:.5f} |\n'
    # Package 205 selected masks: 100 native photos x2 conditions +5 actual known cases.
    exports=root/'mask_exports';exports.mkdir(exist_ok=True);index=[]
    for split,run,method,manifests in [
        ('development',root/'R4_ablations','full',[f'data/campaign100_v3/{co}_explore.jsonl' for co in ('target_proxy','all_shadow_proxy')]),
        ('validation',validation,'mask_raw',[f'data/campaign100_v3/{co}_validation.jsonl' for co in ('target_proxy','all_shadow_proxy')]),
        ('new20',final,'shadow_mask',[f'data/shadow_mask_holdout20/{co}_heldout.jsonl' for co in ('target_proxy','all_shadow_proxy')]),
        ('actual5',validation,'mask_raw',['data/regression_actual5.jsonl'])]:
        for manifest in manifests:
            for r in load_manifest(manifest):
                co=r.get('cohort','actual');src=run/method/co/r['id']/'prediction.png';dest=exports/split/co/f'{r["id"]}.png';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy(src,dest)
                index.append({'split':split,'cohort':co,'id':r['id'],'source_name':r.get('name',r['id']),'mask_path':str(dest.relative_to(exports)),'mask_sha256':sha256(dest),'source_sha256':sha256(r['source']),'generated_sha256':sha256(r['generated']),'ground_truth_sha256':sha256(r['shadow_mask'])})
    (exports/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index));assert len(index)==205
    with zipfile.ZipFile(root/'shadow_masks_205.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in exports.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(exports))
    # Additional actual-only preview: directly visible masks, full five rows.
    preview=Image.new('RGB',(6*230,5*260),'#f4f4ef');draw=ImageDraw.Draw(preview)
    for i,r in enumerate(actual_records):
        s=read_rgb(r['source'])[0];t=read_mask(r['shadow_mask']);p=read_mask(validation/'mask_raw'/'existing_actual_model_candidate'/r['id']/'prediction.png')
        strict=read_mask(validation/'background_alpha'/'existing_actual_model_candidate'/r['id']/'prediction.png')
        values=[(s,r['id']+' 原图'),(read_mask(r['mask']).astype('uint8')*255,'原物体mask'),(t.astype('uint8')*255,'影子真值'),(strict.astype('uint8')*255,'严格回贴区域'),(p.astype('uint8')*255,'纯影子mask'),(overlay(s,t,p),'命中青／误检橙／漏检紫')]
        for j,(a,title) in enumerate(values):
            im=ImageOps.contain(Image.fromarray(a).convert('RGB'),(224,224));preview.paste(im,(j*230+(230-im.width)//2,i*260+30+(224-im.height)//2));draw.text((j*230+4,i*260+5),title,fill='black',font=font(13))
    preview.save(assets/'actual5_preview.png')
    s=summary['all_shadow_proxy'];t=summary['target_proxy'];a=realsummary['shadow_mask']
    sensitivity_text='；'.join(f'{co}: IoU {sensitivity["shadow_mask"][co]["iou"]:.5f}' for co in ('target_proxy','all_shadow_proxy'))
    stability_text='\n'.join(f'- {co}：亮度+2 {v["stability_brightness2"]:.5f}，RGB+[2,-2,1] {v["stability_color2"]:.5f}，噪声σ0.8 {v["stability_noise08"]:.5f}。' for co,v in stability.items())
    boards='\n\n'.join(f'![{NAMES[m]}成功失败与mask输出](assets/board_{m}.png)' for m in METHODS)
    pages='\n'.join(f'- [{p.stem}](assets/{p.name})' for p in compact_paths)
    report_text=f'''# 目标影子纯 mask：IoU 探索

阶段报告。完成本轮选型与冻结评估；复杂场景的目标归属仍未解决。

## 结论

现有真实生成5张开发案例，纯mask平均IoU **{100*a['iou']:.2f}%**，P **{100*a['precision']:.2f}%**，R **{100*a['recall']:.2f}%**；严格回贴区域对照IoU {100*realsummary['background_alpha']['iou']:.2f}%。最高开发替代为平滑版81.30%，因代理验证较差未选用。

新增20张照片的冻结评估：只去目标影子代理IoU **{100*t['iou']:.2f}%**；消除全部影子的代理IoU **{100*s['iou']:.2f}%**。两者输入候选质量及目标归属难度不同，不能混合成单一“检测准确率”，也不构成真实消除模型泛化结果。

只预测mask，因此没有RGB回贴引起的全图改色。mask仍会误覆盖真实非编辑背景，全图去影代理背景误检率{100*s['background_fpr']:.4f}%；不能承诺将该mask用于回贴后背景绝对不变。CPU完整推理平均{timings['shadow_mask']:.1f}ms/张，包含背景补偿，未调用GPU或新增生成模型。

## 实验设置与选择

开发集60张原生DESOBA照片，各有目标代理与全图代理候选；另5个已知真实生成案例。旧20验证照片选择raw与smooth，新增20照片只做冻结评价。原生样本共100张（60+20+20），另5个已知真实生成。原历史20测试未用于本轮选型。输入原图、冻结候选、物体mask；GT只参与开发评分和最终评价，不进入预测函数。代理候选由作者去影参考和标注构造，物体仍保留；真实5图则为历史消除模型候选，输入为256像素拼图分离素材，不是原生全尺寸新测试。

64组粗搜索、16组阈值扩展、4组边界验证（含重复锚点），另4个关闭/放宽模块消融。开发选择三类条件等权macro IoU，验证在raw/smooth间比较两种代理macro IoU均值：raw 0.70811，smooth 0.66175。旧验证曾被历史研究使用，不能当成全新验证。

新20固定seed=20261023，最长边384。文件名分组、pHash、SIFT去重控制照片重复，不能保证同赛事或拍摄事件独立。源图复核在测试前指定排除d017/d020的18图敏感性：{sensitivity_text}。20图主结果与18图敏感性都保留，未依据测试结果改参数。冻结代码提交240bc55。

```mermaid
flowchart LR
 A[原图与冻结消除候选] --> B[稳健背景偏移补偿]
 B --> C[绝对与相对变亮残差]
 D[物体mask] --> B
 D --> E[排除物体并约束目标邻接]
 C --> E
 E --> F[只输出二值影子mask]
 F --> G[推理后与目标影子真值评分]
```

配置：绝对亮度残差>24；相对增亮>0.24（分母最大值为20与原图亮度）；关联半径max(2,0.02×短边)；搜索半径<0.65×短边；无平滑、无弱阈值扩展、无强制核心、无填孔、无羽化。连通分量须满足面积与强证据门槛。内部背景补偿仅用于残差信号，不输出校色图。三个旧回贴预设不改。

## 指标与完整定量比较

TP为预测与真值交集；FP为预测之外非目标投影，包括其他物体影子；FN为漏掉的目标影子。物体mask全部排除。IoU=TP/(TP+FP+FN)，Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，均越大越好、0–1；空预测对非空真值P记0。背景误检率=FP/真实非编辑背景像素数，越小越好，显示为百分比。主结果是逐图宏平均，微平均先累计TP/FP/FN再求IoU，会更多受大影子支配。最差IoU直接揭示完全漏检。

{metrics_table}

![质量、背景误检和完整CPU耗时](assets/metrics.png)

图中IoU为宏平均；完整CPU耗时只来自新20照片×两种候选的同一进程评测，包含预处理/背景拟合，排除磁盘读取、评分和保存。执行有并行进程竞争，不作硬件性能承诺。纯mask控制了误检，但高阈值会漏弱影，距离约束会漏远端影子。

### 已知真实五图逐图结果

{per_image}

![真实五图纯mask预览](assets/actual5_preview.png)

这五张是开发病例。真值和预测均按原分辨率对齐；二值mask不含原物体。预览保留纯黑白输出，误检和漏检由最后一列单独说明。

## 模块消融与稳定性

{ablation}

关目标关联时全图代理IoU明显下降，同时目标代理召回上升：当前目标归属靠空间先验，遇到消除其他影子的候选仍会混淆。放宽弱阈值扩展对真实五图有害；局部低残差并不可靠地代表目标软影。高绝对阈值筛去背景漂移，但不能仅凭亮度判定影子语义。

固定候选扰动与原预测mask的IoU如下（不是对GT的IoU；不用于选型）：

{stability_text}

扰动seed=20261023+行号。空预测与空扰动预测的IoU记1（已修正评分中的空集合约定，未改预测或GT分数）；目标代理有3/20张空预测、全图代理有2/20张空预测，均是漏检，稳定不等于正确。experiments另列非空预测的稳定性。这里只检查小色偏、亮度、噪声，不证明视频时序或运动/压缩鲁棒。GT未进入任何扰动推理。

### 分组结果

{group_table}

分组可以重叠，dark/long/textured等由原数据统计阈值形成，并非人工材质或软硬影标注。

## 逐模型可视化：成功、失败、mask输出

{boards}

### 视觉核验

宽回贴区域常把邻近其他投影一起包含；严格回贴区域缩小范围但仍受物体附近残差影响；纯mask更贴近高对比影子主体，弱影/距离较远处有漏检。新图d014牛群、d020跑步三种方法均完全漏检；d017在仅目标代理下纯mask也为空。已知跑步例r01的物体边缘残余会形成零碎误检，飞机例r03与硬件例r05轮廓贴合更好。每个方法都保留IoU最低的实际评测案例；不因完全漏检而隐藏该图。橙色为误检，紫色为漏检，青色为命中。

## 同帧 compact 对比与输出

每页相同照片、相同候选、相同GT，第一行是输入与各法二值mask，第二行是位置与误检/漏检诊断。完整mask包共205个PNG：100张原生照片×两种代理条件＋5张真实生成案例。PNG白色255为预测影子，黑色0为其他像素，原物体全部黑色；按开发、验证、新测试、真实五图分目录，保留输入与预测SHA清单。

{pages}

## 局限、建议与失败范围

采用冻结纯mask配置作为当前精准定位入口，保留平滑版为开发对照。它适用于候选确实改变了目标影子且背景变化较小的情况；候选未消除影子时残差检测无从识别，候选把其他影子也消掉时难以判断归属。新20的全图代理结果表明，仅靠当前残差与邻接规则还无法稳定得到高IoU目标影子。这是本轮结果的边界，不再按测试案例继续调整阈值。

<details>
<summary>复现细节与产物</summary>

### 命令

```bash
shadow-fusion predict-mask --source input.png --generated edited.png --mask object.png --output outputs/shadow.png --mask-config configs/shadow_mask_selected.json
shadow-fusion predict-mask --source input.png --generated edited.png --mask object.png --output outputs/no_relative.png --mask-config configs/shadow_mask_selected.json --ablate relative
python scripts/explore_shadow_mask.py --manifest data/shadow_mask_holdout20/target_proxy_heldout.jsonl --manifest data/shadow_mask_holdout20/all_shadow_proxy_heldout.jsonl --output outputs/new_run --methods outputs/shadow_mask_20261021/final_methods.json --stability
python -m unittest discover -s tests -v
```

### 环境、输入与冻结

实验环境与精确配置见experiments；Python3.13、NumPy2.3.5、SciPy1.16.3、Pillow12，Apple M1 CPU。DESOBA原生档案SHA256为6abb5d985c0bcce025ad56af505f0d268662e5e8e91aedbf4152f02af845c755。26项单元测试通过，原60回贴输出逐像素回归通过。完整R0–R4开发分数保留在本地实验目录；公开仓库仅收录参数/指标/冻结清单，没有重发第三方原图。

### 产物

[精确最终分数](experiments/final_shadow_mask_metrics.json)、[18图敏感性](experiments/sensitivity18.json)、[稳定性与耗时](experiments/stability_and_timing.json)、[冻结选择](experiments/selection.json)。

SVG图为assets/metrics.svg；mask包位于报告父目录shadow_masks_205.zip。mask本身不对外部图片做修改。相同mask用于回贴时应另行评价真实背景误改与接缝，不能复用本报告当作RGB零误差证明。

</details>
'''
    (report/'report.md').write_text(report_text)
    (exp/'artifact_index.json').write_text(json.dumps({'mask_count':205,'compact_count':len(compact_paths),'files':[{ 'path':str(p.relative_to(report)),'sha256':sha256(p)} for p in assets.glob('*')]},indent=2)+'\n')
    return {'report':str(report/'report.md'),'selected_macro_iou':{'actual_known5':a['iou'],**{c:v['iou'] for c,v in summary.items()}},'masks':205}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);a=p.parse_args();print(json.dumps(build(a.root,a.report),indent=2))
