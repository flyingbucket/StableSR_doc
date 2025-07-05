# pytorch lighting模块简介

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

## pytorch_Lightning Callback 机制

PyTorch Lightning 的 Callback（回调机制）是训练过程中的**事件钩子系统**，允许用户在训练 / 验证 / 测试 / 保存等阶段插入自定义逻辑，类似于钩子（hook）或监听器。

---

### ✅ 常见用途

- 自动保存最佳模型（如根据 val/loss 最小）
- Early stopping（提前停止训练）
- 日志记录 / 学习率可视化
- 自定义日志、评估、样本可视化等行为

---

### 🧩 核心概念：Callback 是一个类

```python
from pytorch_lightning.callbacks import Callback

class MyCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("训练开始！")
    
    def on_validation_end(self, trainer, pl_module):
        print("验证阶段结束。")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f"训练第 {batch_idx} 个 batch 完成")
```

---

### 🧪 使用 Callback 的方式

```python
from pytorch_lightning import Trainer

trainer = Trainer(callbacks=[MyCallback()])
```

可以一次性注册多个 callback：

```python
trainer = Trainer(callbacks=[
    MyCallback(),
    ModelCheckpoint(...),
    EarlyStopping(...)
])
```

---

### 📦 官方内置常用回调

#### 1. `ModelCheckpoint`: 保存最优模型

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",     # 监控哪个指标
    save_top_k=1,           # 只保留 top1
    mode="min",             # 目标是最小化 loss
    filename="best-checkpoint",
    save_last=True
)
```

#### 2. `EarlyStopping`: 提前停止

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor="val/loss",
    patience=5,     # 若 5 个 epoch 没有提升则停止
    mode="min"
)
```

---

### 📚 常用 Callback 钩子方法一览

| 方法名                    | 调用时机            |
| ---------------------- | --------------- |
| `on_fit_start`         | fit() 开始时       |
| `on_train_start`       | 训练阶段开始          |
| `on_train_end`         | 训练阶段结束          |
| `on_train_batch_start` | 每个 batch 开始前    |
| `on_train_batch_end`   | 每个 batch 结束后    |
| `on_validation_end`    | 每轮验证结束后         |
| `on_save_checkpoint`   | 保存 checkpoint 时 |
| `on_load_checkpoint`   | 加载 checkpoint 时 |

---

### 🎯 实例：在验证后记录当前模型状态

```python
class LogModelNorm(Callback):
    def on_validation_end(self, trainer, pl_module):
        total_norm = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"当前梯度范数：{total_norm:.4f}")
```

---

### ✅ 小结

- Callback 是一种轻量级的插件系统；
- 所有模型相关回调都可以集中管理，不污染核心训练逻辑；
- 非常适合记录日志、保存中间结果、动态修改训练行为等。

