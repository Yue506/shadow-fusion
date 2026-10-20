"""Build an evidence-backed local report and all-scene, same-frame compacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask
from run_campaign_round import aggregate
from compact_campaign import compact,font,overlay

METHODS=['mask','tracefusion','shadow','selected']
NAMES={'mask':'原 mask 回贴','tracefusion':'TraceFusion 复现','shadow':'冻结 shadow','selected':'选定配置'}
COLORS=['#A8A8A8','#7884B4','#B4C0E4','#484878']


def read_rows(root):return [json.loads(l) for l in (Path(root)/'progress.jsonl').read_text().splitlines()]
def percent(x):return f'{100*x:.6g}%'
def number(x):return f'{x:.5f}'


def gallery(campaign,data,selection,cohort):
    records=load_manifest(data/f'{cohort}_all.jsonl');assert len(records)==100
    root=campaign/f'gallery_{cohort}';root.mkdir(exist_ok=True)
    rows=[];audit=[]
    for method in METHODS:
        for split,prefix in [('explore','r0'),('validation','val'),('final_test','final')]:
            name=method
            if method=='selected':
                if split=='explore':prefix='selected_explore'
                elif split=='validation':name=selection['selected_validation_id']
            source_root=campaign/f'{prefix}_{cohort}'
            lookup={r['id']:r for r in read_rows(source_root) if r['method']==name}
            for record in (r for r in records if r['split']==split):
                sample=record['id'];src=source_root/name/sample
                assert (src/'output.png').is_file(),src
                dst=root/method/sample;dst.parent.mkdir(exist_ok=True)
                if dst.exists():assert dst.resolve()==src.resolve()
                else:dst.symlink_to(src.resolve(),target_is_directory=True)
                row={**lookup[sample],'method':method};rows.append(row)
                source=read_rgb(record['source'])[0];output=read_rgb(src/'output.png')[0]
                envelope=read_mask(src/'envelope.png');mask=read_mask(record['mask']);target=read_mask(record['shadow_mask'])
                difference=np.abs(output.astype(np.int16)-source.astype(np.int16))
                outside=int((~envelope).sum());outside_max=int(difference[~envelope].max()) if outside else None
                assert outside_max in (0,None)
                background=~(mask|target)
                changed=int(np.any(difference[background]!=0,-1).sum())
                ratio=changed/int(background.sum())
                assert abs(ratio-row['background_changed_fraction'])<1e-12
                audit.append({'id':sample,'method':method,'split':split,'outside_pixels':outside,
                              'outside_max':outside_max,'background_pixels':int(background.sum()),
                              'background_changed_pixels':changed,'output_sha256':hashlib.sha256((src/'output.png').read_bytes()).hexdigest()})
    (root/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'pixel_audit.json').write_text(json.dumps(audit,indent=2))
    (root/'metrics.json').write_text(json.dumps({m:aggregate([r for r in rows if r['method']==m]) for m in METHODS},indent=2))
    assert len(rows)==400 and len(audit)==400
    return root,records,rows,audit


def board(records,root,method,rows,path):
    data=[r for r in rows if r['method']==method]
    best=max(data,key=lambda r:r['shadow_f1']);worst=min(data,key=lambda r:r['shadow_f1'])
    lookup={r['id']:r for r in records};canvas=Image.new('RGB',(1008,336),'#f4f4ef');draw=ImageDraw.Draw(canvas);f=font(16)
    for col,row in enumerate([best,worst,best]):
        r=lookup[row['id']];source=read_rgb(r['source'])[0];target=read_mask(r['shadow_mask'])
        if col==2:a=overlay(source,target,read_mask(root/method/r['id']/'prediction.png'))
        else:a=read_rgb(root/method/r['id']/'output.png')[0]
        im=ImageOps.contain(Image.fromarray(a),(326,260));x=col*336
        title=['相对较好' if method!='mask' else '范围外保护','较差场景','区域诊断'][col]
        draw.text((x+5,4),f'{NAMES[method]}｜{title}',fill='black',font=f)
        canvas.paste(im,(x+(336-im.width)//2,34+(260-im.height)//2))
        draw.text((x+5,303),f'{r["id"]}  F1 {row["shadow_f1"]:.3f}  背景 {percent(row["background_changed_fraction"])}',fill='black',font=f)
    canvas.save(path)


def build(campaign,data,report,selection_file):
    campaign=Path(campaign);data=Path(data);report=Path(report);assets=report/'assets';exp=report/'experiments'
    assets.mkdir(exist_ok=True);exp.mkdir(exist_ok=True)
    selection=json.loads(Path(selection_file).read_text());assert selection['frozen_before_test']
    final={co:json.loads((campaign/f'final_{co}'/'metrics.json').read_text()) for co in ['target_proxy','all_shadow_proxy']}
    for payload in final.values():assert payload['count']==20 and payload['plan_sha256']==selection['plan_sha256']
    galleries={};final_rows={}
    for co in final:
        root,records,rows,audit=gallery(campaign,data,selection,co);galleries[co]=(root,records,rows,audit)
        final_rows[co]=read_rows(campaign/f'final_{co}')
        compact(data/f'{co}_all.jsonl',[str(root/m) for m in METHODS],assets/f'all_{co}',per_page=4,selection=selection_file)
        shutil.copyfile(root/'pixel_audit.json',exp/f'pixel_audit_{co}.json')
        shutil.copyfile(campaign/f'final_{co}'/'metrics.json',exp/f'final_{co}.json')
        shutil.copyfile(campaign/f'final_{co}'/'progress.jsonl',exp/f'final_{co}_progress.jsonl')
        shutil.copyfile(root/'progress.jsonl',exp/f'all_{co}_progress.jsonl')
    # Eight cases include quality extremes, median cases and instability/background extremes.
    co='all_shadow_proxy';root,records,rows,audit=galleries[co]
    selected=[r for r in final_rows[co] if r['method']=='selected'];ordered=sorted(selected,key=lambda r:r['shadow_f1'])
    proposed=[*ordered[:2],*ordered[-2:],*ordered[9:11],max(selected,key=lambda r:r['background_changed_fraction']),min(selected,key=lambda r:r['stability_iou_mean'])]
    ids=list(dict.fromkeys(r['id'] for r in proposed))
    for r in ordered:
        if len(ids)>=8:break
        if r['id'] not in ids:ids.append(r['id'])
    compact(data/f'{co}_final_test.jsonl',[str(campaign/f'final_{co}'/m) for m in METHODS],assets/'overview',ids=','.join(ids),selection=selection_file)
    test_records=load_manifest(data/f'{co}_final_test.jsonl')
    for m in METHODS:board(test_records,campaign/f'final_{co}',m,final_rows[co],assets/f'board_{m}.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ['/System/Library/Fonts/STHeiti Light.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if Path(name).exists():
            font_manager.fontManager.addfont(name);plt.rcParams['font.family']=font_manager.FontProperties(fname=name).get_name();break
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for j,(metric,title,scale) in enumerate([('shadow_f1','目标投影 F1 ↑',1),('background_changed_fraction','真实背景误改（%）↓',100),('stability_iou_mean','小扰动区域 IoU ↑',1)]):
        for i,(co,label) in enumerate([('target_proxy','仅目标代理'),('all_shadow_proxy','全图去影代理')]):
            values=[final[co]['methods'][m]['summary'][metric]*scale for m in METHODS]
            axes[j].bar(np.arange(4)+(i-.5)*.36,values,.34,label=label,color=COLORS,alpha=1 if i==0 else .45,hatch=None if i==0 else '//',edgecolor='#484878')
        axes[j].set_xticks(range(4),['原mask','TraceFusion','冻结shadow','选定配置'],rotation=15);axes[j].set_title(title);axes[j].grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=9);fig.suptitle('锁定测试：每种代理条件 20 张；柱高为逐场景宏平均',fontsize=14)
    fig.tight_layout();fig.savefig(assets/'metrics.png',dpi=180);fig.savefig(assets/'metrics.svg');plt.close(fig)
    headers='| 候选条件 | 方法 | 精确率 | 召回率 | F1 | IoU | 最差 F1 | 背景误改 | 背景零改图数 | 扰动 IoU | 最差扰动 IoU |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    table=headers
    rng=np.random.default_rng(42);intervals={}
    for co,label in [('target_proxy','仅目标代理'),('all_shadow_proxy','全图去影代理')]:
        for m in METHODS:
            s=final[co]['methods'][m]['summary'];rs=[r for r in final_rows[co] if r['method']==m]
            zeros=sum(r['background_changed_fraction']==0 for r in rs)
            table+=f'| {label} | {NAMES[m]} | {s["shadow_precision"]:.5f} | {s["shadow_recall"]:.5f} | {s["shadow_f1"]:.5f} | {s["shadow_iou"]:.5f} | {min(r["shadow_f1"] for r in rs):.5f} | {percent(s["background_changed_fraction"])} | {zeros}/20 | {s["stability_iou_mean"]:.5f} | {min(r["stability_iou_min"] for r in rs):.5f} |\n'
        base={r['id']:r for r in final_rows[co] if r['method']=='shadow'}
        delta=np.array([r['shadow_f1']-base[r['id']]['shadow_f1'] for r in final_rows[co] if r['method']=='selected'])
        draws=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(-1)
        intervals[co]={'paired_f1_gain':float(delta.mean()),'bootstrap95':np.quantile(draws,[.025,.975]).tolist(),'scenes':20,'seed':42}
    (exp/'bootstrap.json').write_text(json.dumps(intervals,indent=2))
    group_names={'dark':'暗背景代理','long':'长影代理','textured':'纹理背景代理','weak_or_soft':'弱／软影代理','strong_or_hard':'强／硬影代理','neighbor_shadow':'邻近其他投影','multi_object':'多物体'}
    group_table='| 条件 | 分组 | 图数 | F1 | 召回 | 背景误改 | 扰动 IoU |\n|---|---|---:|---:|---:|---:|---:|\n'
    for co,label in [('target_proxy','目标代理'),('all_shadow_proxy','全图代理')]:
        for group,s in final[co]['methods']['selected']['groups'].items():
            group_table+=f'| {label} | {group_names.get(group,group)} | {s["count"]} | {s["shadow_f1"]:.5f} | {s["shadow_recall"]:.5f} | {percent(s["background_changed_fraction"])} | {s["stability_iou_mean"]:.5f} |\n'
    # Matched ablations on validation only, never used to retune the frozen selection.
    ablation=''
    for co,label in [('target_proxy','目标代理'),('all_shadow_proxy','全图代理')]:
        path=campaign/f'ablate_{co}'/'metrics.json'
        if path.exists():
            m=json.loads(path.read_text());shutil.copyfile(path,exp/f'ablate_{co}.json')
            for name,v in m['methods'].items():
                s=v['summary'];ablation+=f'| {label} | {name} | {s["shadow_f1"]:.5f} | {s["shadow_recall"]:.5f} | {percent(s["background_changed_fraction"])} | {s["stability_iou_mean"]:.5f} |\n'
    actual='';p=campaign/'final_actual_regression'/'metrics.json'
    if p.exists():
        m=json.loads(p.read_text());shutil.copyfile(p,exp/'actual_regression.json')
        for name,v in m['methods'].items():
            s=v['summary'];actual+=f'| {NAMES[name]} | {s["shadow_f1"]:.5f} | {s["shadow_recall"]:.5f} | {percent(s["background_changed_fraction"])} | {s["shadow_reference_rgb_l1_255"]:.5f} |\n'
    cfg=selection['selected_config'];summary=final['all_shadow_proxy']['methods']['selected']['summary']
    boards='\n'.join(f'![{NAMES[m]}代表与失败案例](assets/board_{m}.png)\n\n{NAMES[m]}：'+{'mask':'范围外保护完全，但不识别投影。','tracefusion':'目标覆盖高，候选色偏和其他投影容易被一并回贴。','shadow':'校色和有限羽化缩小改动范围，仍有背景误检。','selected':'置信度抑制无证据拷贝；分离弱影和邻近其他投影仍是失败来源。'}[m] for m in METHODS)
    text=f'''# 100 张影子编辑区域：稳定性与背景保护

**阶段报告｜真实非编辑背景逐像素不变尚未达成。**

- 100 张原生场景已按 seed=42 冻结为 60 探索、20 验证、20 锁定测试；完成 R0–R8 九轮探索及匹配消融。每张有两种冻结代理候选，不是 200 张独立照片。
- 验证规则选中 `{selection['selected_validation_id']}`。锁定测试的全图去影条件中，F1={summary['shadow_f1']:.5f}，召回={summary['shadow_recall']:.5f}，背景误改={percent(summary['background_changed_fraction'])}。范围外数字像素保护通过，语义背景保护仍会失败。
- 百图是作者参考构造的代理，只评价回贴机制；物体仍保留于参考中。五张已有真实生成候选另做已知案例回归，不能把代理成绩当成消除模型泛化。
- 最轻的原 mask 回贴没有额外区域开销，但不能去除投影。对象检测和局部同色拟合未带来足够综合收益，保留为显式探索模块。

## 数据与推理

数据来自 [DESOBA 官方入口](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA) 的原生档案。图像级去重包括文件名变体、感知哈希、几何匹配和源图复核；每张仅选一个物体投影关联。最长边不超过 384，保留纵横比，原生图仍保留在下载档案。原档案分组为 83 Training、16 Testing、1 Remaining；这是自定义困难场景集，不是官方榜单分割。

目标代理仅替换目标投影；全图代理使用作者整图去影参考，因此包含本应保留的其他投影改动。两者统一注入固定小色偏、空间平面和弱噪声。构造可以使用投影标注，**推理接口只接收原图、冻结候选、用户物体 mask、统一配置**。标注在推理及扰动推理后才评分。无新增付费生成或 GPU 任务。

```mermaid
flowchart TD
 A[原图、冻结候选、原物体 mask] --> B[背景锚点稳健校色]
 B --> C[正向亮度与纹理证据]
 C --> D[可选空间关联与置信度]
 D --> E[有限羽化与编辑域内回贴]
 A --> F[编辑域外逐像素复制原图]
 E --> G[无损 PNG 与区域图]
 F --> G
 H[投影及保护背景标注] --> I[仅评估：区域、背景、稳定性]
 G --> I
```

选择配置：校色 `{cfg['calibration']}`，关联 `{cfg['association']}`，置信度 `{cfg['confidence']}`（{cfg['gate_low']}→{cfg['gate_high']}），羽化 `{cfg['feather']}`，gamma `{cfg['gamma']}`。完整参数已在测试前提交冻结，未按测试结果调参。

## 锁定测试的精确结果

区域预测为 mask 外低频 alpha>0.5。精确率=TP/(TP+FP)，召回=TP/(TP+FN)，F1、IoU按每场景计算后宏平均。非编辑背景为物体和目标投影之外的全部像素，含其他物体及其投影；任一 RGB 通道变化至少 1 即计为误改。范围外相等只检验预测域外，不代表语义判断正确。稳定性为候选统一 +2、RGB+[2,-2,1]、固定 sigma=0.8 噪声三种扰动与原区域的 IoU；不是视频时间一致性。

{table}
![锁定测试指标](assets/metrics.png)

选定配置相对冻结 shadow 的配对 F1 增量与场景重采样 95% 区间：目标代理 {intervals['target_proxy']['paired_f1_gain']:.5f}，{intervals['target_proxy']['bootstrap95']}；全图代理 {intervals['all_shadow_proxy']['paired_f1_gain']:.5f}，{intervals['all_shadow_proxy']['bootstrap95']}。这是固定二十场景的描述性不确定性，不能替代真实生成候选验证。

## 同帧 compact 与失败案例

前两列为原图和指定物体 mask，第三列是冻结去影代理，随后是四组回贴；第二行给出投影标注及 TP/FP/FN 诊断。mask 可能指定背景小物体，画面中央其他物体的投影应受保护。八例由最好、最差、中位与不稳定场景共同选出，未据此修改配置。

![同帧对照一](assets/overview/compact_001.png)
![同帧对照二](assets/overview/compact_002.png)

{boards}

## 匹配模块消融（验证集，每条件 20 张）

选定完整配置后仅切换对应模块，保持候选不变；用于解释取舍，不再用于修改锁定配置。

| 条件 | 消融 | F1 | 召回 | 背景误改 | 扰动 IoU |
|---|---|---:|---:|---:|---:|
{ablation}

## 真实生成候选的已知五图回归

这五张来自前期实际生成，不包含在新百图中，且已用于开发诊断，不能算独立泛化测试。只统计投影恢复与背景保护，物体消除效果由图像定性审阅。

| 方法 | F1 | 召回 | 背景误改 | 投影参考 RGB 误差 |
|---|---:|---:|---:|---:|
{actual}

## 解释与限制

选定配置的分组表现如下。分组相互重叠，人数不相加；弱／软、强／硬、材质相关词均指量化代理，而非人工语义标注。

{group_table}

稳健校色与正向置信度是本轮主要收益；直接对整帧残差羽化会把小色偏和其他投影一起回贴。gamma 在初始四组中主要增加背景，缩进羽化则严重损失狭窄投影。弱路径改善部分分离区域，仍不能判对所有投影归属。CPU YOLOX 框竞争错误分走了目标投影；局部同色拟合对本轮已知平面扰动几乎无收益，增加了耗时。

代理候选与其已知扰动有明确构造机制，尤其有利于匹配的校色模型；结果不是百图生成质量。二十张锁定测试较小，组标签是亮度、纹理、距离等量化代理，未提供独立人工材质／软硬影标注。低对比、远端投影、相邻投影合并及生成器真实改错内容仍会失败。数字像素保证只针对无损 PNG 与同尺寸解码 RGB；真实背景完全不改的质量目标没有达成。

<details>
<summary>复现、全部 compact 与证据索引</summary>

探索分支：[Yue506/shadow-fusion](https://github.com/Yue506/shadow-fusion/tree/explore/shadow-regions-20261021)，冻结基线 `baseline-20261021`。输入档案 SHA、100 图清单、分割、逐图数据和每轮源代码提交均在仓库 `docs/results/campaign_20261021/`。

```bash
python scripts/rebuild_frozen_campaign.py --archive data/desoba/DESOBA_DATASET.zip --output data/campaign100
shadow-fusion compose --source input.png --generated edited.png --mask mask.png --output outputs/result.png --research-config configs/selected.json
shadow-fusion compose --source input.png --generated edited.png --mask mask.png --output outputs/no_confidence.png --research-config configs/selected.json --ablate confidence
```

两条件各 25 页、每页四场景：`assets/all_target_proxy/` 与 `assets/all_all_shadow_proxy/`。本地全尺寸输出与区域图仍保留在 campaign 各运行目录，gallery 连接对应来源。全部 800 组输出的范围外及真实背景像素已重新核验，详见 [目标条件审计](experiments/pixel_audit_target_proxy.json) 和 [全图条件审计](experiments/pixel_audit_all_shadow_proxy.json)。原始照片与生成候选不再公开分发，公开仓库只保留代码、哈希及数值证据。

[目标测试原始指标](experiments/final_target_proxy.json)｜[全图测试原始指标](experiments/final_all_shadow_proxy.json)｜[配对区间](experiments/bootstrap.json)

CPU 耗时是本机墙钟，探索轮次可能存在同机并行竞争；如提供串行复测，以复测文件为准。没有实测消除模型推理耗时，因此不报告后处理相对模型的百分比加速或开销。未使用重置券或新窗口继续研究。

</details>
'''
    (report/'report.md').write_text(text)
    shutil.copyfile(selection_file,exp/'selection.json')
    (exp/'artifact_index.json').write_text(json.dumps({'cohorts':list(final),'primary_count':20,'all_scene_count':100,'overview_ids':ids,'methods':METHODS},indent=2))
    shutil.copyfile(data/'dataset.json',exp/'dataset.json')
    metadata_path=report/'report_metadata.json'
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(status='stage report',profile='model-benchmark',quality_target_met=False,
                    new_scenes=100,split={'explore':60,'validation':20,'final_test':20},
                    exploratory_rounds=9,proxy_conditions=2,selected_validation_id=selection['selected_validation_id'])
    metadata_path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    archive=campaign/'shadow_campaign_compact_report.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in report.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(report.parent)))
    print(json.dumps({'report':str(report/'report.md'),'archive':str(archive),'all_pages':50,'audit_records':800},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--data',required=True)
    p.add_argument('--report',required=True);p.add_argument('--selection',default='docs/results/campaign_20261021/selection.json')
    a=p.parse_args();build(a.campaign,a.data,a.report,a.selection)
