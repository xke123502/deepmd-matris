# `is_train` 参数改动说明

## 为什么需要 `is_train` 参数？

### 问题背景
在使用预计算图（`use_precomputed_graphs=True`）时，需要区分：
- **Training数据**：使用 `train_systems` 列表中的路径
- **Validation数据**：使用 `valid_systems` 列表中的路径

DeepMD的训练流程中，同一个模型会同时处理training和validation数据，但它们的system路径不同。如果不区分，validation时会错误地从training的路径加载图，导致：
1. 路径错误（validation数据在`filtered_val`，但可能加载`filtered_train`的图）
2. `sid`索引错误（validation的`sid`应该索引`valid_systems`，而不是`train_systems`）

### 解决方案
通过 `is_train` 参数明确标识当前是training还是validation阶段，让模型根据这个标志选择正确的systems列表。

---

## 改动详细说明

### 1. `wrapper.py` 改动

#### 原始版本（无 `is_train`）
```python
def forward(
    self,
    coord,
    atype,
    # ... 其他参数
    # 没有 sid, fid, is_train
):
    input_dict = {
        "coord": coord,
        "atype": atype,
        # ...
    }
    # 直接调用模型，不传递额外参数
    model_pred = self.model[task_key](**input_dict)
```

#### 修改版本（添加 `is_train`）
```python
def forward(
    self,
    coord,
    atype,
    # ... 其他参数
    sid: Optional[int] = None,        # ← 新增：system id
    fid: Optional[list] = None,        # ← 新增：frame id列表
    is_train: Optional[bool] = True,   # ← 新增：是否为training
):
    input_dict = {
        "coord": coord,
        "atype": atype,
        # ...
    }
    
    # ← 新增：动态检查模型是否支持这些参数
    # 只在模型支持时传递 sid、fid 和 is_train (added in 2025-11-13)
    import inspect
    model_forward = self.model[task_key].forward
    forward_params = inspect.signature(model_forward).parameters
    if "sid" in forward_params:
        input_dict["sid"] = sid
    if "fid" in forward_params:
        input_dict["fid"] = fid
    if "is_train" in forward_params:  # ← 关键：检查模型是否支持is_train
        input_dict["is_train"] = is_train
    
    model_pred = self.model[task_key](**input_dict)
```

**改动原因**：
- 使用 `inspect` 动态检查，确保向后兼容（不支持这些参数的模型不会报错）
- `is_train` 会被传递到模型的 `forward` 方法

---

### 2. `training.py` 改动

#### 改动点1：Training时传递 `is_train=True`

**原始版本**（第729-730行）：
```python
model_pred, loss, more_loss = self.wrapper(
    **input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key
)
```

**修改版本**（第752-753行）：
```python
model_pred, loss, more_loss = self.wrapper(
    **input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key, is_train=True  # ← 新增
)
```

#### 改动点2：Validation时传递 `is_train=False`

**原始版本**（第845-850行）：
```python
_, loss, more_loss = self.wrapper(
    **input_dict,
    cur_lr=pref_lr,
    label=label_dict,
    task_key=_task_key,
)
```

**修改版本**（第868-873行）：
```python
_, loss, more_loss = self.wrapper(
    **input_dict,
    cur_lr=pref_lr,
    label=label_dict,
    task_key=_task_key,
    is_train=False,  # ← 新增：明确标识这是validation
)
```

#### 改动点3：Multi-task情况下的改动

**原始版本**（第919-924行）：
```python
_, loss, more_loss = self.wrapper(
    **input_dict,
    cur_lr=pref_lr,
    label=label_dict,
    task_key=_key,
)
```

**修改版本**（第919-924行）：
```python
_, loss, more_loss = self.wrapper(
    **input_dict,
    cur_lr=pref_lr,
    label=label_dict,
    task_key=_key,
    is_train=True,  # ← 新增：multi-task training时
)
```

**改动原因**：
- Training时：`is_train=True` → 模型使用 `train_systems` 列表
- Validation时：`is_train=False` → 模型使用 `valid_systems` 列表

---

### 3. `model_e0.py` 改动

#### 改动点1：`forward` 方法接收 `is_train`

```python
@torch.jit.export
def forward(
    self,
    coord: torch.Tensor,
    atype: torch.Tensor,
    # ... 其他参数
    sid: Optional[int] = None,
    fid: Optional[list[int]] = None,
    is_train: Optional[bool] = True,  # ← 新增：接收is_train参数
) -> dict[str, torch.Tensor]:
    # ...
    if self.use_precomputed_graphs and sid is not None and fid is not None:
        # 加载预计算的图，传递is_train
        batch_graph = self._load_precomputed_graphs(sid, fid, is_train=is_train)  # ← 传递is_train
```

#### 改动点2：`_load_precomputed_graphs` 根据 `is_train` 选择systems

```python
def _load_precomputed_graphs(self, sid: int, fid: list[int], is_train: bool = True):
    """
    加载预计算的graphs
    
    参数:
        sid: system id（相对于 training 或 validation 的索引）
        fid: frame ids列表
        is_train: 是否为 training（True）或 validation（False）  # ← 关键参数
    """
    # 根据 is_train 选择对应的 systems 列表
    if is_train:
        systems = self.train_systems  # ← Training时使用train_systems
    else:
        systems = self.valid_systems  # ← Validation时使用valid_systems
    
    if sid >= len(systems):
        raise IndexError(f"sid {sid} 超出范围，systems 数量: {len(systems)}, is_train: {is_train}")
    
    system_path = systems[sid]  # ← 从正确的列表中选择路径
    # ...
```

**改动原因**：
- `is_train=True` → 使用 `self.train_systems[sid]` → 加载training数据的图
- `is_train=False` → 使用 `self.valid_systems[sid]` → 加载validation数据的图
- 确保 `sid` 索引的是正确的systems列表

---

## 数据流示意

```
Training阶段:
  training.py (step函数)
    → get_data(is_train=True)  # 获取training数据
    → wrapper.forward(..., is_train=True)  # 传递is_train=True
      → model_e0.forward(..., is_train=True)
        → _load_precomputed_graphs(sid, fid, is_train=True)
          → 使用 self.train_systems[sid]  # ✅ 正确路径

Validation阶段:
  training.py (log_loss_valid函数)
    → get_data(is_train=False)  # 获取validation数据
    → wrapper.forward(..., is_train=False)  # 传递is_train=False
      → model_e0.forward(..., is_train=False)
        → _load_precomputed_graphs(sid, fid, is_train=False)
          → 使用 self.valid_systems[sid]  # ✅ 正确路径
```

---

## 关键点总结

1. **向后兼容**：使用 `inspect` 检查模型是否支持 `is_train`，不支持的模型不会报错
2. **明确区分**：通过 `is_train` 明确标识training/validation阶段
3. **路径正确**：确保validation时从 `valid_systems` 加载图，而不是 `train_systems`
4. **索引正确**：`sid` 索引的是对应阶段的systems列表，不会越界或错位

---

## 为什么需要这么改？

**如果不改**：
- Validation时，`sid` 会索引 `train_systems`（因为默认使用training的systems）
- 导致路径错误：`filtered_val/...` 的数据却加载 `filtered_train/...` 的图
- 导致索引错误：validation的 `sid=0` 可能对应 `train_systems[0]`，而不是 `valid_systems[0]`

**改了之后**：
- Validation时，`is_train=False` → 使用 `valid_systems[sid]`
- 路径正确：validation数据加载validation的图
- 索引正确：`sid` 索引的是 `valid_systems`，不会错位

