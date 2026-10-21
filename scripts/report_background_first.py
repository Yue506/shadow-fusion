"""Background-first report from immutable run metrics and same-scene outputs."""
import argparse,json,shutil,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from shadow_fusion.benchmark import load_manifest
from shadow_fusion.io import read_rgb,read_mask,sha256
from compact_campaign import compact,font,overlay
from audit_background_first import audit

METHODS=['prior_selected','conservative','candidate']
NAMES={'prior_selected':'前轮代理选定','conservative':'保守单级','candidate':'背景优先候选'}


def build(root,report):
    root=Path(root);report=Path(report);assets=report/'assets';exp=report/'experiments'
    assets.mkdir(exist_ok=True);exp.mkdir(exist_ok=True)
    selection_path=Path('docs/results/background_first_20261021/candidate_test_freeze.json');selection=json.loads(selection_path.read_text())
    assert selection['frozen_before_test'];tables=[];summaries={};audits={}
    primary='| 条件（各20张） | 方法 | 背景误改≥1 | 背景误改>3 | 背景RGB误差 | 背景高频误差 | 最差背景误改 |\n|---|---|---:|---:|---:|---:|---:|\n'
    secondary='| 条件 | 方法 | 召回 | F1 | 投影RGB误差 | 内边界RGB | 外边界RGB | 边界梯度误差 | 扰动IoU | 回贴范围 | CPU毫秒 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for co,label in [('target_proxy','目标代理'),('all_shadow_proxy','全图去影代理')]:
        manifest=f'data/background_first_holdout20/{co}_final_test.jsonl'
        assert sha256(manifest)==selection['holdout_manifest_sha256'][co]
        run=root/f'final_{co}';raw=json.loads((run/'metrics.json').read_text());assert raw['count']==20 and raw['plan_sha256']==selection['plan_sha256']
        ap=root/f'audit_final_{co}.json'
        if not ap.exists():audit(manifest,run,ap)
        a=json.loads(ap.read_text());audits[co]=a;summaries[co]=a['summary']
        for method in METHODS:
            s=a['summary'][method]
            primary+=f'| {label} | {NAMES[method]} | {s["background_changed_fraction"]*100:.5f}% | {s["background_changed_over3_fraction"]*100:.5f}% | {s["background_rgb_l1_255"]:.5f} | {s["background_highpass_l1_255"]:.5f} | {s["background_changed_fraction_max"]*100:.5f}% |\n'
            secondary+=f'| {label} | {NAMES[method]} | {s["shadow_recall"]:.5f} | {s["shadow_f1"]:.5f} | {s["shadow_reference_rgb_l1_255"]:.5f} | {s["shadow_inner_boundary_rgb_l1_255"]:.5f} | {s["protected_outer_boundary_rgb_l1_255"]:.5f} | {s["target_boundary_gradient_l1_255"]:.5f} | {s["stability_iou_mean"]:.5f} | {s["envelope_fraction"]*100:.3f}% | {s["cpu_seconds"]*1000:.2f} |\n'
        compact(manifest,[str(run/m) for m in METHODS],assets/co,selection=str(selection_path))
        shutil.copyfile(ap,exp/f'audit_{co}.json');shutil.copyfile(run/'metrics.json',exp/f'final_{co}.json')
    # Method boards use identical heldout condition, and retain genuine failures.
    co='all_shadow_proxy';run=root/f'final_{co}';lookup={r['id']:r for r in load_manifest(f'data/background_first_holdout20/{co}_final_test.jsonl')}
    for method in METHODS:
        rows=[r for r in audits[co]['rows'] if r['method']==method]
        best=max(rows,key=lambda r:r['shadow_f1']);worst=max(rows,key=lambda r:r['background_changed_fraction'])
        canvas=Image.new('RGB',(1008,336),'#f4f4ef');draw=ImageDraw.Draw(canvas);f=font(15)
        for col,row in enumerate([best,worst,best]):
            r=lookup[row['id']];source=read_rgb(r['source'])[0];target=read_mask(r['shadow_mask'])
            a=overlay(source,target,read_mask(run/method/r['id']/'prediction.png')) if col==2 else read_rgb(run/method/r['id']/'output.png')[0]
            im=ImageOps.contain(Image.fromarray(a),(326,260));x=col*336
            draw.text((x+5,4),NAMES[method]+'｜'+['相对成功','背景失败','区域诊断'][col],fill='black',font=f)
            canvas.paste(im,(x+(336-im.width)//2,34+(260-im.height)//2))
            draw.text((x+5,303),f'{r["id"]} F1 {row["shadow_f1"]:.3f} 背景 {row["background_changed_fraction"]*100:.2f}%',fill='black',font=f)
        canvas.save(assets/f'board_{method}.png')
    actual=root/'final_actual';ap=root/'audit_final_actual.json'
    if not ap.exists():audit('data/regression_actual5.jsonl',actual,ap)
    actual_data=json.loads(ap.read_text());shutil.copyfile(ap,exp/'actual_known5.json')
    compact('data/regression_actual5.jsonl',[str(actual/m) for m in METHODS],assets/'actual',per_page=3)
    actual_table='| 已知真实五图 | 背景误改 | >3级误改 | 背景RGB | 投影召回 | 投影RGB | 边界梯度 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in METHODS:
        s=actual_data['summary'][m];actual_table+=f'| {NAMES[m]} | {100*s["background_changed_fraction"]:.4f}% | {100*s["background_changed_over3_fraction"]:.4f}% | {s["background_rgb_l1_255"]:.5f} | {s["shadow_recall"]:.5f} | {s["shadow_reference_rgb_l1_255"]:.5f} | {s["target_boundary_gradient_l1_255"]:.5f} |\n'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp=Path('/System/Library/Fonts/STHeiti Light.ttc')
    if fp.exists():font_manager.fontManager.addfont(str(fp));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams['svg.fonttype']='none'
    colors=['#A8A8A8','#7884B4','#484878'];fig,axs=plt.subplots(1,4,figsize=(16,4))
    for ax,(metric,title,scale) in zip(axs,[('background_changed_fraction','背景误改（%）↓',100),('shadow_reference_rgb_l1_255','投影RGB误差↓',1),('shadow_recall','投影召回↑',1),('cpu_seconds','CPU墙钟毫秒↓',1000)]):
        for i,co in enumerate(['target_proxy','all_shadow_proxy']):
            ax.bar(np.arange(3)+(i-.5)*.36,[summaries[co][m][metric]*scale for m in METHODS],.34,color=colors,alpha=1 if i==0 else .45,hatch='' if i==0 else '//',label=['目标代理','全图去影代理'][i])
        ax.set_title(title);ax.set_xticks(range(3),['前轮选定','保守单级','背景优先'],rotation=15);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(assets/'metrics.png',dpi=160);fig.savefig(assets/'metrics.svg');plt.close(fig)
    b=summaries['all_shadow_proxy']['conservative'];s=summaries['all_shadow_proxy']['candidate']
    real=actual_data['summary']['candidate'];boards='\n\n'.join(f'![{NAMES[m]}成功、失败与输出](assets/board_{m}.png)' for m in METHODS)
    report_text=f'''# 背景保护优先：第二轮回贴探索

阶段报告；真实非编辑区域绝对零改仍未达成。

## 结论

- 新策略未通过预定验证召回线，规则回退为 `{selection['selected_id']}`。下面另行评价冻结的 `{selection['candidate_id']}` 候选，不能称为验证胜者。新增二十场景、全图去影代理的背景误改由保守单级 {100*b['background_changed_fraction']:.4f}% 降至 {100*s['background_changed_fraction']:.4f}%；投影召回 {s['shadow_recall']:.5f}，投影 RGB 误差 {s['shadow_reference_rgb_l1_255']:.5f}。
- 已知真实五图背景误改 {100*real['background_changed_fraction']:.4f}%，投影召回 {real['shadow_recall']:.5f}，投影 RGB 误差 {real['shadow_reference_rgb_l1_255']:.5f}；旧病例是开发回归，不是独立泛化证据。
- 严格光度守卫会漏掉几乎全部真实投影；仅保留原纹理的亮度迁移显著恶化去影颜色，因此均不作为默认。窄羽化及相对亮度门控呈现背景保护与漏影的取舍。
- 新二十张来自 DESOBA 原生档案，排除旧100与已知5；两种候选均由作者参考构造。没有新增消除模型调用或 GPU 开销。

## 实验设置

先在旧60探索图和已知真实5图比较 R0–R4 五组假设，再按预先规则使用复用20验证图选择。旧20主测试没有再次使用。新增20场景以固定 seed=20261022，源图文件名、pHash及几何匹配去重；场景事件级身份仍不可保证。候选、分辨率最长边384、色偏及噪声构造沿用冻结协议。

```mermaid
flowchart LR
 A[原图、候选、用户mask] --> B[稳健背景锚点校色]
 B --> C[亮度证据与目标空间关联]
 C --> D[绝对及相对置信度]
 D --> E[有限羽化并局部回贴]
 A --> F[范围外复制原图]
 E --> G[无损输出]
 F --> G
 G --> H[推理结束后才用投影标注评分]
```

## 首要指标：真实标注背景保护

背景为用户物体及目标投影之外全部像素，含其他物体与投影。任一RGB通道差≥1计误改，>3另列；均逐图宏平均。RGB及高频误差为0–255标尺的通道绝对差，高频去除sigma=1高斯低频。它们是像素诊断，不能把高频误差直接等同人类内容判断。

{primary}

![质量与计算记录](assets/metrics.png)

## 次要指标：颜色、范围与边界

影子以作者去影参考评分。召回与F1基于mask外alpha>0.5，不代表已彻底去影；范围为实际有限回贴域占全图比例。内/外边界是目标投影内外两像素，外边界以原图为目标；梯度误差比较Sobel/8与严格参考回贴，排除原物体周围一像素。扰动IoU仅覆盖+2亮度、RGB偏移与sigma=.8噪声；此前压缩与错位敏感性未因此自动解决。CPU是本机推理墙钟，不含消除生成，轮次耗时不作为速度优化结论。

{secondary}

## 已知真实候选回归

{actual_table}

![真实五图compact一](assets/actual/compact_001.png)
![真实五图compact二](assets/actual/compact_002.png)

## 逐模型可视化

{boards}

## 视觉核验与失败模式

前轮代理选定方案容易误接其他投影和背景残差；保守单级减少部分错误，但外扩范围较大。背景优先候选降低低相对亮度改动和边界外扩，同时会漏掉低对比、分离投影。案例板保留相对成功及背景误改最大的失败例，青/橙/紫为命中/误判/漏检。原mask内仍保留校色候选，无法从本数据证明物体消除后的语义内容正确。

## 同帧对比与 compact

![新增场景对比一](assets/all_shadow_proxy/compact_001.png)
![新增场景对比二](assets/all_shadow_proxy/compact_002.png)

两条件各五页展示全部20场景；其余页位于同目录。每页四图，列为原图、物体mask、冻结候选及三组输出，第二行是区域诊断。代理保留原物体，不用于评价物体消除质量。

## 取舍与限制

规则先要求两条件召回≥0.8、扰动IoU≥0.95及投影RGB误差不超保守基线1.2倍加1；再最小化较差条件的背景误改。在5%相对或0.05百分点工程带内考虑投影RGB、边界梯度及耗时。规则不代表统计等价。新20图测试前配置已提交，未按其结果改参数。

这轮降低了部分错误回贴，仍不能识别所有语义背景。作者参考代理与真实模型生成存在明显差异，旧5例也不能代替独立真实候选集。校色/边界真值只针对投影和原图背景；物体消除区域无独立真值。JPEG、视角错位、复杂彩色投影仍是风险。原有预设保持不变，新策略显式指定配置。

<details>
<summary>复现与完整证据</summary>

分支：[explore/background-first-20261021](https://github.com/Yue506/shadow-fusion/tree/explore/background-first-20261021)。冻结参数：`configs/background_first_candidate.json`。全部有效配置、逐图指标和数据清单位于公开仓库 `docs/results/background_first_20261021/`；原生照片、候选和完整输出只在本地保存。

```bash
python scripts/rebuild_frozen_campaign.py --archive data/desoba/DESOBA_DATASET.zip --output data/holdout20 --selection docs/results/background_first_20261021/heldout_selection.jsonl --metadata docs/results/background_first_20261021/heldout_dataset.json
shadow-fusion compose --source input.png --generated edited.png --mask mask.png --output outputs/result.png --research-config configs/background_first_candidate.json
shadow-fusion compose --source input.png --generated edited.png --mask mask.png --output outputs/no_confidence.png --research-config configs/background_first_candidate.json --ablate confidence
```

测试与原60张冻结输出回归已通过。模块可以独立关闭；实验失败的guard、luminance、luma_offset仍只作为复现消融入口。GT仅在推理后用于评分。`experiments/`保留精确数值与选择规则，`assets/`保留PNG和可编辑SVG。

</details>
'''
    report_text=report_text.replace('新增20场景','新增20张图片').replace('新增二十场景','新增二十张图片').replace('新二十张来自 DESOBA 原生档案，排除旧100与已知5','新二十张来自 DESOBA 原生档案，自动筛除旧100及已知5的照片近重复')
    report_text=report_text.replace('梯度误差比较Sobel/8与严格参考回贴，排除原物体周围一像素。','梯度诊断在RGB三维数组上沿空间轴计算Sobel/8，含通道平滑，只作本实现内对照；与严格参考回贴比较，排除原物体周围一像素。')
    c=actual_data['summary']['conservative'];r=actual_data['summary']['candidate']
    boundary_note=f'真实五图中，候选的外边界RGB误差 {c["protected_outer_boundary_rgb_l1_255"]:.3f}→{r["protected_outer_boundary_rgb_l1_255"]:.3f}，但边界梯度诊断 {c["target_boundary_gradient_l1_255"]:.3f}→{r["target_boundary_gradient_l1_255"]:.3f}，投影RGB误差 {c["shadow_reference_rgb_l1_255"]:.3f}→{r["shadow_reference_rgb_l1_255"]:.3f}；不能宣称校色和接缝全面改善。\n\n'
    report_text=report_text.replace('## 逐模型可视化',boundary_note+'## 逐模型可视化')
    event_path=root/'event_sensitivity19.json'
    if event_path.exists():
        event=json.loads(event_path.read_text());shutil.copyfile(event_path,exp/'event_sensitivity19.json')
        note='## 数据复核与19图敏感性\n\n视觉复核发现d009与旧r01来自同一赛事的不同照片，pHash距离24，自动去重未排除事件关联。保留冻结20图主结果，不能称20个独立新场景。排除d009仅作事后敏感性分析，不替换样本或修改参数；其他事件身份也无完整元数据。\n\n| 19图条件 | 方法 | 背景误改 | 召回 | 投影RGB误差 |\n|---|---|---:|---:|---:|\n'
        for co,label in [('target_proxy','目标代理'),('all_shadow_proxy','全图代理')]:
            for m,s in event['summary'][co].items():note+=f'| {label} | {NAMES[m]} | {100*s["background_changed_fraction"]:.5f}% | {s["shadow_recall"]:.5f} | {s["shadow_reference_rgb_l1_255"]:.5f} |\n'
        report_text=report_text.replace('## 取舍与限制',note+'\n## 取舍与限制')
    (report/'report.md').write_text(report_text)
    shutil.copyfile(selection_path,exp/'selection.json')
    meta=json.loads((report/'report_metadata.json').read_text());meta.pop('new_holdout_scenes',None)
    meta.update(status='stage report',profile='model-benchmark',quality_target_met=False,new_test_photos=20,known_event_related_photos=1,posthoc_sensitivity_photos=19,event_independence_guaranteed=False,selected_id=selection['selected_id'])
    (report/'report_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    with zipfile.ZipFile(root/'background_first_compact_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in report.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(report.parent)))
    print(report/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--report',required=True);a=p.parse_args();build(a.campaign,a.report)
