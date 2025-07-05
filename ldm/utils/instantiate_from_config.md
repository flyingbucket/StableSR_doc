```python
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

```

## 作用
读取config配置文件,按照其指定规则将target指定的类实例化.同时在第2行if块中给出了不进行实例化的判断逻辑,可以利用它来控制开关某些模块(如注意力机制和结构条件).
## config 配置要求
应该使用yaml文件,包含target项用于指定类名,params项及其子项用于指定实例化时的模型参数.

## 具体的实例化逻辑和过程
[[get_obj_from_str]] 返回一个类对象,在get_obj_from_str(...)后面的 `(**config.get)("params",dict())`相当于将params作为参数传入到前面的类对象的__init__中,最终完成实例化.
