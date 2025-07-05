```python
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config) # [[instantiate_from_config]]
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, struct_cond=None, seg_cond=None):
        if self.conditioning_key is None: # [[#None：无条件扩散（Unconditional）]]
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat': # [[#concat：通道拼接（Channel Concat）]]
            xc = torch.cat([x] + c_concat, dim=1) 
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn': # [[#crossattn：交叉注意力（Cross-Attention）]]
            cc = torch.cat(c_crossattn, 1)
            if seg_cond is None:
                out = self.diffusion_model(x, t, context=cc, struct_cond=struct_cond)
            else:
                out = self.diffusion_model(x, t, context=cc, struct_cond=struct_cond, seg_cond=seg_cond)
        elif self.conditioning_key == 'hybrid': # [[#hybrid：通道拼接 + 交叉注意力（Concat + Cross-Attn）]]
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm': # [[#adm：标签注入（Classifier-free Guidance）]]
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out

```

## `DiffusionWrapper.forward()` 中 conditioning_key 的五种模式解析

在 `StableSR` 或 `Latent Diffusion` 中，`conditioning_key` 决定了条件信息如何注入到扩散模型（通常是 UNet，如 `UNetModelDualConv2d`）中。该参数影响的是 `DiffusionWrapper` 在 forward 阶段如何组织输入，并通过哪些通路将条件信息传递到扩散网络。

---

### None：无条件扩散（Unconditional）

```python
out = self.diffusion_model(x, t)
```

- 不使用任何条件信息；
- 输入仅为带噪图像 `x` 和时间步 `t`；
- 通常用于纯图像建模、初期训练或 unconditional generation。

---

### concat：通道拼接（Channel Concat）

```python
xc = torch.cat([x] + c_concat, dim=1)
out = self.diffusion_model(xc, t)
```

- 条件以图像形式提供（如低清图、边缘图、小波子带），直接拼接到 `x` 上；
- 通道维度变为 `C + C_cond`；
- 是 SR3 使用的典型条件注入方法；
- 是传统sr3的方法,简单高效，但灵活性较差。

---

###  crossattn：交叉注意力（Cross-Attention）

```python
cc = torch.cat(c_crossattn, 1)
out = self.diffusion_model(x, t, context=cc, struct_cond=..., seg_cond=...)
```

- 条件信息（如结构图、小波图、文本）通过 `cond_stage_model` 编码为 latent 向量；
- 这些 latent 向量作为 `context`，传入 UNet 内部的 `CrossAttention` 模块；
- 可选地支持结构条件 `struct_cond` 和语义条件 `seg_cond`；
- 是现代条件扩散（如 Stable Diffusion）最常用策略，灵活且表达力强。

---

### hybrid：通道拼接 + 交叉注意力（Concat + Cross-Attn）

```python
xc = torch.cat([x] + c_concat, dim=1)
cc = torch.cat(c_crossattn, 1)
out = self.diffusion_model(xc, t, context=cc)
```

- 同时使用 concat 和 cross-attention 两种通路注入条件；
- 通道拼接传递低级信息（细节、边缘）；
- cross-attn 传递高级语义（结构 latent、小波 latent）；
- 在 StableSR 中，通常两路条件信息都来源于同一个结构图，只是经过不同路径处理；
- 平衡引导性与灵活性，是推荐模式之一。

---

### adm：标签注入（Classifier-free Guidance）

```python
cc = c_crossattn[0]
out = self.diffusion_model(x, t, y=cc)
```

- 用于类别或 token 作为标签条件（如 ImageNet class label）；
- `y=cc` 表示将条件作为类标签注入；
- 需要扩散模型支持 `y` 输入（如通过 ConditionalBatchNorm、embedding 等）；
- 常见于 ADM/DDPMv2 等分类条件生成场景。

---

## 总结

| conditioning_key | 条件注入方式   | 典型用途                             |
| ---------------- | -------- | -------------------------------- |
| `None`           | 无条件      | 纯图像建模                            |
| `'concat'`       | 通道拼接     | SR3、结构图引导                        |
| `'crossattn'`    | 交叉注意力    | Stable Diffusion、结构/文本引导         |
| `'hybrid'`       | 拼接 + 注意力 | StableSR、小波 + 结构联合引导             |
| `'adm'`          | 标签输入     | 分类条件生成（如 class-conditional DDPM） |

实际应用中，可根据条件类型和任务目标灵活选择或组合这些模式。
