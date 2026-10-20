# 模块与接口约定

`compose(source, generated, mask, make_config(...))` 接受同尺寸 `uint8 RGB` 原图／候选和非空布尔物体 mask，不读取评估标签，不加载模型。

```python
from shadow_fusion import compose, make_config
result = compose(source, generated, mask,
                 make_config('shadow', gamma=True, feather='multilevel', ablate=('color',)))
output = result['output']
```

| 模块 | 实际作用 | 关闭后的行为 |
|---|---|---|
| `color` | 在可靠背景对应位置稳健拟合 linear RGB 每通道增益＋偏置 | 直接使用原始候选颜色 |
| `extra` | 检测原物体 mask 之外的编辑支持 | 只保留模式对应的物体核心，gamma 同时关闭 |
| `gamma` | 对原图与候选共同做 γ=0.75，补充暗部变化支持 | 不运行补充分支 |
| `feather` | 支持边缘过渡；可选择有限、论文式、多级 | 支持内硬回贴，范围外复制原图 |

`mask` 核心就是原 mask；`tracefusion` 和 `shadow` 的核心分别沿用各自原先的轻微膨胀。`--ablate extra` 不等于移除这些核心膨胀。额外 RGB 校色与原 TraceFusion 检测中的背景中位数残差补偿是不同操作。

原 TraceFusion gamma 接在原残差检测之后。`shadow` 的 gamma 接在校色后的正亮度／纹理主检测之后，保留原论文暗区、邻接和面积门槛，并沿用本原型的最大搜索距离。因此两个模式都是明确可追踪的配置，但不能把 `shadow` 叫作完整原算法。

多级羽化分别融合细、中、低频；诊断 `alpha` 是低频 alpha。支持域内部始终完整使用校色候选。最终范围外显式复制原图，但若范围覆盖全图，则没有受保护像素，元数据返回 `protected_pixel_count=0`、`outside_envelope_max_channel_difference=null`，不伪造“背景保护通过”。

CLI 输入要求：8-bit RGB 图像和同尺寸二值 mask（0/1 或 0/255，L 或等值 RGB 通道）。不自动缩放，不隐式对齐，不处理 HDR、透明图或不同色彩空间匹配。输出使用 PNG，并沿用原图 ICC 描述；数字像素保证是相对于已解码输入 RGB。JPEG／有损视频编码后的像素变化不在该保证内。

JSON 只支持五个配置键：`mode`、`color`、`extra`、`gamma`、`feather`。未知模块、空 mask、错尺寸、非二值 mask 或覆盖输入的输出路径都会报错。批量运行要求新建输出目录，防止将不同版本的结果混在一起。
