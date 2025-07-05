# init 函数介绍

```python
class DDPM(pl.LightningModule): # [[pytorch lightning#pl.LightningMoudule]]
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__() 
        # [[#parameterization]]
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key) # [[DiffusionWrapper(pl.LightningModule)]]
        count_params(self.model, verbose=True) # [[#line 45 to 57 🔧 `DDPM.__init__` 中模型管理、调度器与损失权重部分解析]]
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None: # [[#📈 PyTorch Lightning 中的 `monitor` 参数简析]]
            self.monitor = monitor 
        if ckpt_path is not None: # 加载预训练模型
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet) # [[init_from_ckpt]]
		# [[#line 64 to end]]
		# [[#1. 注册调度表（beta schedule）]]
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
		# [[#2. 设置损失函数类型]]
        self.loss_type = loss_type
		# [[#logvar 在 DDPM 扩散模型中的作用与实现]]
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

```



## parameterization

- `parameterization`: 扩散模型的预测目标，有三种：
    - `"eps"`: 噪声预测（最常用）
    - `"x0"`: 预测原始图像
    - `"v"`: v-pred 方案，平衡两个极端
- 断言限制只支持上述三种模式

## line 45 to 57 🔧 `DDPM.__init__` 中模型管理、调度器与损失权重部分解析

这部分代码负责扩散模型训练过程中的几个重要功能组件的配置，包括参数统计、EMA 平滑、调度器设置，以及损失项的加权策略。

---

### 🔢 模型参数统计与打印

```python
count_params(self.model, verbose=True)
```

- 调用 `count_params` 打印模型参数数量；
- `self.model` 是之前创建的 `DiffusionWrapper`（包含 UNet 和条件控制）；
- `verbose=True` 表示输出详细层级参数统计，有助于模型调试与规模评估。

---

### 🧮 EMA 模型配置（Exponential Moving Average）

```python
self.use_ema = use_ema
if self.use_ema:
    self.model_ema = LitEma(self.model)
    print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
```

- `use_ema`: 控制是否启用 EMA；
- 如果启用，将创建 `model_ema`，用于在训练中对模型参数进行滑动平均；
- EMA 可在推理时提供更稳定的结果（尤其训练后期）；
- `LitEma` 是一个内部实现的 EMA 工具类（模仿 PyTorch EMA 实现）；
- `buffers()` 提供了所有被 EMA 追踪的张量（通常是模型权重）。

PS. [EMA](https://zhuanlan.zhihu.com/p/68748778)简介
指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），是一种给予近期数据更高权重的平均方法。


### 📈 训练调度器配置（如学习率调度）

```python
self.use_scheduler = scheduler_config is not None
if self.use_scheduler:
    self.scheduler_config = scheduler_config
```

- 如果提供了 `scheduler_config`，则将其保存；
- 后续在 `configure_optimizers()` 方法中会用到该配置；
- 可用于定义如余弦退火（cosine annealing）、线性 warmup 等学习率调度策略。

---

### ⚖️ 损失项权重设置

```python
self.v_posterior = v_posterior
self.original_elbo_weight = original_elbo_weight
self.l_simple_weight = l_simple_weight
```

这三项参数控制扩散损失的组成：

| 参数名                    | 含义                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `v_posterior`          | 后验方差的加权控制项。用于设置预测方差的策略。具体计算形式为：<br>`σ² = (1 - v) * beta_tilde + v * beta`，<br>其中 `v` 就是这个参数。 |
| `original_elbo_weight` | 是否加入原始论文中的 ELBO loss 项（常为 0，代表不启用）                                                           |
| `l_simple_weight`      | L2 或 L1 损失的主权重，用于稳定训练                                                                        |

- 这部分最终将作用于 `get_loss()` 或 `p_losses()` 中的损失函数组合；
- 对于不同任务（如图像重建 vs 生成），可以通过调整这些参数来平衡生成质量和保真度。

---


### ✅ 小结

这一部分为训练过程提供了必要的配置管理：
- **模型参数统计**有助于可视化规模；
- **EMA 管理**可提升训练稳定性；
- **调度器设置**为优化器行为提供了灵活性；
- **损失项加权**控制生成模型优化目标的侧重点。


## 📈 PyTorch Lightning 中的 `monitor` 参数简析

`monitor` 是 PyTorch Lightning 中回调（Callback）机制的一部分，用于指定**训练过程中要监控的指标名称**，供如 `ModelCheckpoint`、`EarlyStopping` 等回调依据该指标执行相应逻辑（如保存模型、提前停止等）。
详见[[pytorch lightning#pytorch_Lightning Callback 机制]]

---

### ✅ 关键用途

| 组件                | 用途             |
| ----------------- | -------------- |
| `ModelCheckpoint` | 保存性能最好的模型      |
| `EarlyStopping`   | 在验证指标停止提升时中止训练 |

---

### 🧩 monitor 的工作流程

1. **模型内部记录指标**：
   ```python
   self.log("val/loss", val_loss, prog_bar=True)
   ```

2. **指定 monitor 的回调监听这个指标**：
   ```python
   ModelCheckpoint(monitor="val/loss", mode="min")
   ```

3. **Lightning 自动比较并触发保存 / 停止逻辑**。

---

### ⚙️ monitor 示例配置

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="val/psnr",   # 监听 PSNR 指标
    mode="max",           # 指标越大越好
    save_top_k=1,
    filename="best-psnr"
)
```

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val/loss",
    mode="min",
    patience=5
)
```

---

### 🧠 说明

- `monitor` 是一个字符串，必须和 `.log(...)` 中记录的名字一致；
- 不会自动创建指标值，只是引用已有指标；
- 和 `mode` 一起决定何时触发操作（`min` → 越小越好，`max` → 越大越好）；
- 在模型类中可用 `self.monitor` 传递给 callback 实现动态配置。

---

### ✅ 总结

- `monitor` 是 **回调系统监听的指标名称**；
- 配合 `.log(...)` 使用；
- 决定是否保存模型 / 提前停止；
- 本身不计算指标，仅作为引用字段使用。


## line 64 to end

这一部分主要完成了噪声调度与损失配置,具体来说,
本部分代码完成了扩散过程中的**beta调度表构建**(即噪声调度)、**损失函数类型设定**和**对数方差的初始化**，是 DDPM 建模核心参数的关键配置步骤。

---

### 1. 注册调度表（beta schedule）

```python
self.register_schedule(
    given_betas=given_betas,
    beta_schedule=beta_schedule,
    timesteps=timesteps,
    linear_start=linear_start,
    linear_end=linear_end,
    cosine_s=cosine_s
)
```

- 该函数根据 `beta_schedule` 的类型（如 "linear" 或 "cosine"）构建扩散过程中的时间步噪声系数表；
- 通常会生成如下参数：
  - `betas`：每一步的噪声幅度；
  - `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod` 等；
- 这些系数会在后续 `q_sample`, `q_posterior`, `predict_start_from_noise` 等函数中使用；
- 参数说明：
  - `linear_start`, `linear_end`: 用于线性 beta 调度起止值；
  - `cosine_s`: 用于调整余弦调度的形状；
  - `given_betas`: 若提供，优先使用用户自定义 beta 表。

---

### 2. 设置损失函数类型

```python
self.loss_type = loss_type
```

- 控制训练时使用哪种损失：
  - `"l2"`（默认）：预测的噪声与真实噪声之间的均方误差；
  - `"l1"`：预测残差的绝对值损失；
  - 也可能支持其他自定义损失类型，如 perceptual loss、hybrid loss 等；
- 实际使用在 `get_loss()` 或 `p_losses()` 中处理。

---

### 3. 初始化对数方差（log-variance）

```python
self.learn_logvar = learn_logvar
self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
if self.learn_logvar:
    self.logvar = nn.Parameter(self.logvar, requires_grad=True)
```

- `logvar` 是一个长度为 `num_timesteps` 的张量，表示每一扩散时间步的对数方差；
- 如果启用 `learn_logvar=True`，则将其设为可学习参数（`nn.Parameter`），允许模型自动优化每一时间步的不确定性；
- 如果不启用，`logvar` 就是一个固定的常量值张量；
- 在 `get_loss()` 中会参与 KL 项或 likelihood 的权重调整。
#### logvar 在 DDPM 扩散模型中的作用与实现

`logvar`（对数方差）是扩散模型（如 DDPM）中用于控制训练损失权重和不确定性建模的一个重要变量。在 StableSR 和 DDPM 实现中，它是 diffusion 模型的一部分，而非 encoder 或 decoder 的组成部分。

---

##### 1. 背景：为何需要 logvar

扩散模型训练时通常以预测噪声为目标，基本损失形式为(以L2损失为例)：
$$
L_t = \left\| \varepsilon_{\text{pred}} - \varepsilon_{\text{true}} \right\|^2
$$
为了增强灵活性、稳定性或逼近对数似然，一些变体引入了对数方差 logvar，使得损失函数变为：
$$
L_t = \frac{1}{2} \cdot \exp(-\text{logvar}_t) \cdot \left\| \varepsilon_{\text{pred}} - \varepsilon_{\text{true}} \right\|^2 + \frac{1}{2} \cdot \text{logvar}_t
$$
这相当于使用一个可变的时间步权重项，用于：
- 控制每一时间步损失的相对重要性；
- 模拟高斯似然中的分布不确定性；
- 使得模型对某些时间步预测更加稳健。

---

##### 2. 初始化方式

logvar 通常被初始化为常数张量：

```python
self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
```

- `logvar_init`: 对数方差的初始值（常为 0）；
- `num_timesteps`: 扩散总步数（如 1000）；
- 得到形状为 `(T,)` 的 logvar 张量，其中 T 是时间步数。

若启用可学习方差：

```python
if self.learn_logvar:
    self.logvar = nn.Parameter(self.logvar, requires_grad=True)
```

则该张量在训练中会自动更新，每一时间步都有不同的可学习不确定性。

---

##### 3. 使用方式（训练时）

logvar 通常参与损失函数定义，在 `get_loss()` 或 `p_losses()` 中用作动态权重：

```python
loss = weighted_mse / torch.exp(self.logvar[t]) + self.logvar[t]
```

或更复杂的：
```python
loss = loss_weight * loss_raw + offset * logvar[t]
```

---

##### 4. logvar总结

| 项目    | 内容                             |
| ----- | ------------------------------ |
| 名称    | logvar（对数方差）                   |
| 类型    | Tensor / nn.Parameter          |
| 维度    | `(num_timesteps,)`             |
| 用途    | 控制不同时间步的损失权重与不确定性建模            |
| 初始化方法 | `torch.full(size, fill_value)` |
| 是否可训练 | 由 `learn_logvar` 参数控制          |

logvar 是一个扩散过程中的权重调节器，尤其在加入 ELBO、VLB 等目标时尤为关键。


---

## 总结

| 项目                        | 功能说明                            |
| ------------------------- | ------------------------------- |
| `register_schedule()`     | 构造扩散过程中每一步的 beta 参数，用于控制加噪过程    |
| `loss_type`               | 决定训练时的损失函数类型，如 L2 或 L1          |
| `learn_logvar` 和 `logvar` | 控制是否学习每个时间步的对数方差，以适配不同的不确定性建模策略 |

这些设置构成了扩散模型训练阶段的核心数学基础。







