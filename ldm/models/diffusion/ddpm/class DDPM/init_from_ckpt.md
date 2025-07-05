```python
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print('<<<<<<<<<<<<>>>>>>>>>>>>>>>')
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
```

## init_from_ckpt(path, ignore_keys=list(), only_model=False) 函数解析

该函数用于从 checkpoint 文件中加载模型参数，可选择只加载 UNet（`self.model`），或加载整个 `DDPM` 模型本身。支持忽略部分参数键名，适用于微调、迁移学习等场景。

---

### 函数签名

```python
def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
```

- `path`: checkpoint 文件路径（通常为 `.ckpt` 或 `.pth`）
- `ignore_keys`: 字符串列表，指定哪些 key 应从 state_dict 中排除（常用于忽略不兼容的模块）
- `only_model`: 若为 True，则只加载 `self.model` 的参数，不加载整个 DDPM 类结构（适用于仅更新 UNet 时）

---

### 1. 加载 checkpoint 字典

```python
sd = torch.load(path, map_location="cpu")
if "state_dict" in list(sd.keys()):
    sd = sd["state_dict"]
```

- 使用 `torch.load` 加载权重；
- 有些 checkpoint 是通过 PyTorch Lightning 保存的，外层是个字典，内部的模型权重位于 `state_dict` 键下；
- 这一步兼容这两种结构。

---

### 2. 根据 ignore_keys 删除不需要的参数

```python
keys = list(sd.keys())
for k in keys:
    for ik in ignore_keys:
        if k.startswith(ik):
            print("Deleting key {} from state_dict.".format(k))
            del sd[k]
```

- 遍历所有参数名（key），如果以 `ik` 中任一字符串开头，则删除该 key；
- 典型用途：跳过 `cond_stage_model`, `model_ema`, `scheduler` 等与当前任务无关的部分。

---

### 3. 加载权重到模型中

```python
missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
```

- `strict=False`：允许 checkpoint 和当前模型结构不完全一致（否则会报错）；
- `missing`：当前模型中存在而 checkpoint 中没有的 key；
- `unexpected`：checkpoint 中存在但当前模型没有的 key；
- 根据 `only_model` 决定加载整个 DDPM 模型或仅加载其 `self.model`（通常是 UNet）。

---

### 4. 打印加载结果

```python
print('<<<<<<<<<<<<>>>>>>>>>>>>>>>')
print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
if len(missing) > 0:
    print(f"Missing Keys: {missing}")
if len(unexpected) > 0:
    print(f"Unexpected Keys: {unexpected}")
```

- 清晰地汇报恢复情况；
- 若 missing/unexpected 过多，可能说明模型结构不匹配，需要调整配置或 `ignore_keys`。

---

### 总结表格

| 参数                       | 作用                           |
| ------------------------ | ---------------------------- |
| `path`                   | 指定加载的 checkpoint 文件路径        |
| `ignore_keys`            | 忽略掉具有特定前缀的参数键（如 `model_ema`） |
| `only_model`             | 是否只加载 `self.model`（通常为 UNet） |
| `strict=False`           | 宽松加载，不要求结构完全一致               |
| `missing` / `unexpected` | 分别记录缺失和多余的参数 key 名           |

此函数广泛用于 StableSR/LDMS/LatentDiffusion 的预训练模型加载与微调流程中，推荐配合 YAML 配置与 callback 一起使用。