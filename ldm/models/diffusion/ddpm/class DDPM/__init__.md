# init 函数介绍

```python
class DDPM(pl.LightningModule): # [[#pl.LightningMoudule]]
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
        count_params(self.model, verbose=True)
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

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

```

## pl.LightningMoudule 

`pl.LightningModule` 是对 `torch.nn.Module` 的高级封装
只需要实现下面几个关键方法，它就能帮你自动完成训练流程、GPU 分发、日志记录、checkpoint保存等复杂操作：

- 1. `__init__(self)`: 初始化模型结构、损失函数、超参数等。
    
- 2. `forward(self, x)`: 定义**前向传播**逻辑（注意：仅在调用 `model(x)` 时使用，**训练逻辑用 `training_step`**）。
    
- 3. `training_step(self, batch, batch_idx)`:定义**一个训练步骤**的行为，返回 `loss`。
	- 通常的步骤是
	    - 从 `batch` 取出数据；
	    - 前向传播；
	    - 计算损失；
	    - 使用 `self.log(...)` 自动记录日志（如 loss）。
        
- 4. `validation_step(...)` / `test_step(...)`:验证和测试阶段的行为，结构与 `training_step` 类似。
    
- 5. `configure_optimizers(self)`:返回优化器和（可选的）学习率调度器。


## parameterization

- `parameterization`: 扩散模型的预测目标，有三种：
    - `"eps"`: 噪声预测（最常用）
    - `"x0"`: 预测原始图像
    - `"v"`: v-pred 方案，平衡两个极端
- 断言限制只支持上述三种模式













