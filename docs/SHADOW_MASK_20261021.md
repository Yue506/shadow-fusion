# 目标影子纯 mask 探索

本分支新增 `predict-mask`，输入原图、冻结消除候选、物体 mask；输出只包含预测影子的二值 PNG。没有物体保护圈、羽化或 RGB 回贴。仍保留背景偏移拟合作为残差检测的预处理，不改变输入 RGB 文件。

```bash
shadow-fusion predict-mask --source input.png --generated edited.png --mask object.png --output outputs/shadow.png --mask-config configs/shadow_mask_selected.json
shadow-fusion predict-mask --source input.png --generated edited.png --mask object.png --output outputs/no_relative.png --mask-config configs/shadow_mask_selected.json --ablate relative
```

可消融 `calibration,relative,association,smoothing,hysteresis,texture`。API 为 `shadow_fusion.shadow_mask.predict_shadow(source, generated, object_mask, config)`，不接受真值或参考图。证据分数不等于经过校准的影子概率。

开发：旧60图片的两类代理候选＋已知真实生成5例；64组粗搜、16组阈值扩展、4组边界消融。三类条件等权平均 macro IoU 决定开发优先级，再用旧20验证图片的两类代理 IoU 均值在 raw/smooth 间选型。旧验证集曾被历史研究使用，不视为全新独立验证。

冻结配置：sRGB空间平面补偿、绝对正亮度>24、相对增亮>0.24、目标关联半径0.02×短边、无平滑、无弱阈值扩展、无纹理过滤。此配置在真实已知5例 IoU 80.81%，平滑替代81.30%；前者旧代理验证均值70.81%，后者66.17%，因此选前者。不能将这5张开发数据叫泛化测试。

新增20不同照片（seed=20261023）在上述配置冻结后一次性评估，不依据其结果再调参。自动照片去重不保证事件独立；预先指定另报告排除d017/d020的18图敏感性。候选是作者参考/标注构造的代理，物体仍在，不是新的消除模型生成。旧60开发＋旧20验证＋新20测试合计100张原生图片，另有已知真实生成5例。

数值、冻结清单与配置见 `docs/results/shadow_mask_20261021/`；完整原图、mask、预测和compact保留本地忽略目录。冻结三个原回贴预设不改，26项测试与原60输出逐像素回归通过。

## 冻结测试与交付结果

| 条件 | 严格回贴区域 IoU | 纯mask IoU | 纯mask Precision | 纯mask Recall |
|---|---:|---:|---:|---:|
| 已知真实生成5张（开发） | 0.51456 | 0.80813 | 0.82960 | 0.96767 |
| 新20：目标去影代理 | 0.70241 | 0.75367 | 0.85000 | 0.75367 |
| 新20：全图去影代理 | 0.39743 | 0.47909 | 0.51153 | 0.79168 |

主指标逐图宏平均；目标代理微平均precision为1（3张完全漏检），不能以它掩盖宏平均召回不足。全图代理真实背景误检1.09737%，仍会混入其他影子。CPU完整推理均值81.64ms（384最长边）。零预测、弱影/远端影漏检和目标归属错误均保留。

冻结推理代码提交`240bc55`后完成新20评估，未按其结果改配置。另修正空mask稳定性评分为IoU(empty,empty)=1，不改预测和对GT的IoU；审计见`final_stability_empty_set_correction.json`。

本地提供205张无损mask与12页同帧compact（100张原生图×2代理＋5张真实生成）。mask不包含原物体；仍依赖冻结消除候选，不是仅凭原图运行的语义影子模型。
