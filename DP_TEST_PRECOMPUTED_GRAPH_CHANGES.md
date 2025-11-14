# dp --pt test 支持预计算图的改动列表

**日期**: 2025-11-14  
**目的**: 让 `dp --pt test` 命令支持使用预计算的图（precomputed graphs）

## 改动文件列表

### 1. `/aisi/mnt/data_nas/jwzhou/opt/deepmd-matris/deepmd_matris/model_e0.py`

#### 改动 1.1: `set_systems` 方法签名和实现
- **位置**: 第 339 行
- **改动类型**: 方法签名修改 + 实现扩展
- **原代码**:
```python
def set_systems(self, train_systems: list[str], valid_systems: list[str] = None):
```
- **新代码**:
```python
def set_systems(self, train_systems: list[str] = None, valid_systems: list[str] = None, test_systems: list[str] = None):
```
- **说明**: 
  - 所有参数改为可选（添加 `= None`）
  - 新增 `test_systems` 参数
  - 实现中改为条件设置（`if train_systems is not None:` 等）
  - 添加 `test_systems` 的设置逻辑（第 365-369 行）

#### 改动 1.2: `_load_precomputed_graphs` 方法签名和实现
- **位置**: 第 371 行
- **改动类型**: 方法签名修改 + 实现扩展
- **原代码**:
```python
def _load_precomputed_graphs(self, sid: int, fid: list[int], is_train: bool = True):
```
- **新代码**:
```python
def _load_precomputed_graphs(self, sid: int, fid: list[int], is_train: bool = True, is_test: bool = False):
```
- **说明**:
  - 新增 `is_test` 参数
  - 在 systems 选择逻辑中添加 `is_test` 分支（第 391-396 行）
  - 更新注释说明（第 385-390 行）

#### 改动 1.3: `forward` 方法签名和调用
- **位置**: 第 425 行（参数）和第 443 行（调用）
- **改动类型**: 方法签名修改 + 调用修改
- **原代码**:
```python
is_train: Optional[bool] = True,  # ========== 改动：新增参数，从wrapper传递过来 (new added in 2025-11-14) ==========
```
```python
batch_graph = self._load_precomputed_graphs(sid, fid, is_train=is_train)  # ← 传递is_train
```
- **新代码**:
```python
is_train: Optional[bool] = True,  # ========== 改动：新增参数，从wrapper传递过来 (new added in 2025-11-14) ==========
is_test: Optional[bool] = False,  # ========== 改动：新增参数，用于测试 (new added in 2025-11-14) ==========
```
```python
batch_graph = self._load_precomputed_graphs(sid, fid, is_train=is_train, is_test=is_test)  # ← 传递is_train和is_test
```
- **说明**:
  - 新增 `is_test` 参数
  - 在调用 `_load_precomputed_graphs` 时传递 `is_test` 参数

---

### 2. `/aisi/mnt/data_nas/jwzhou/deepmd-kit/deepmd/pt/infer/deep_eval.py`

#### 改动 2.1: `__init__` 方法 - 添加属性
- **位置**: 第 172 行
- **改动类型**: 新增属性
- **原代码**: 无
- **新代码**:
```python
self._test_systems = None  # 将在测试时设置
self._current_test_sid = None  # 当前测试的 system id (new added in 2025-11-14)
```
- **说明**: 添加两个属性用于测试模式

#### 改动 2.2: `_eval_model` 方法 - 传递测试参数
- **位置**: 第 467-493 行
- **改动类型**: 新增代码块
- **原代码**:
```python
batch_output = model(
    coord_input,
    type_input,
    box=box_input,
    do_atomic_virial=do_atomic_virial,
    fparam=fparam_input,
    aparam=aparam_input,
)
```
- **新代码**:
```python
# ========== 改动：支持预计算图时传递 sid、fid 和 is_test (new added in 2025-11-14) ==========
# ... (详细注释)
import inspect
model_forward = model.forward
forward_params = inspect.signature(model_forward).parameters
model_kwargs = {
    "coord": coord_input,
    "atype": type_input,
    "box": box_input,
    "do_atomic_virial": do_atomic_virial,
    "fparam": fparam_input,
    "aparam": aparam_input,
}
# 如果模型支持 sid、fid、is_test，并且我们在测试模式，尝试传递这些参数
if "sid" in forward_params and self._current_test_sid is not None:
    model_kwargs["sid"] = self._current_test_sid
if "fid" in forward_params and self._current_test_sid is not None:
    # 假设 fid 是顺序的（0, 1, 2, ...），这可能不准确
    model_kwargs["fid"] = list(range(nframes))
if "is_test" in forward_params:
    model_kwargs["is_test"] = self._current_test_sid is not None

batch_output = model(**model_kwargs)
```
- **说明**: 
  - 使用 `inspect` 动态检查模型是否支持 `sid`、`fid`、`is_test`
  - 如果支持且处于测试模式，传递这些参数
  - 将直接调用改为使用 `model_kwargs` 字典

---

### 3. `/aisi/mnt/data_nas/jwzhou/deepmd-kit/deepmd/entrypoints/test.py`

#### 改动 3.1: `test` 函数 - 设置 test_systems
- **位置**: 第 122-126 行
- **改动类型**: 新增代码块
- **原代码**:
```python
# init model
dp = DeepEval(model, head=head)

for cc, system in enumerate(all_sys):
```
- **新代码**:
```python
# init model
dp = DeepEval(model, head=head)

# ========== 改动：支持预计算图时设置 test_systems (new added in 2025-11-14) ==========
# 目的：让模型知道测试的system路径，以便加载预计算图
# 如果模型支持 set_systems 方法，设置 test_systems
if hasattr(dp.dp.model["Default"], 'set_systems'):
    dp.dp.model["Default"].set_systems(test_systems=all_sys)

for cc, system in enumerate(all_sys):
```
- **说明**: 在初始化模型后，设置测试 systems 列表

#### 改动 3.2: `test` 函数 - 设置当前测试的 system id
- **位置**: 第 132-135 行
- **改动类型**: 新增代码块
- **原代码**:
```python
for cc, system in enumerate(all_sys):
    log.info("# ---------------output of dp test--------------- ")
    log.info(f"# testing system : {system}")

    # create data class
```
- **新代码**:
```python
for cc, system in enumerate(all_sys):
    log.info("# ---------------output of dp test--------------- ")
    log.info(f"# testing system : {system}")

    # ========== 改动：设置当前测试的 system id (new added in 2025-11-14) ==========
    # 目的：让模型知道当前测试的 system id，以便加载预计算图
    if hasattr(dp, '_current_test_sid'):
        dp._current_test_sid = cc

    # create data class
```
- **说明**: 在每个 system 测试前，设置当前测试的 system id

---

## 回退步骤

### 方法 1: 使用 Git 回退（推荐）

如果这些改动已经提交到 Git：

```bash
# 查看改动
git log --oneline --grep="2025-11-14" --grep="test.*precomputed" --all

# 回退到改动前的提交
git revert <commit_hash>
```

### 方法 2: 手动回退

按照以下顺序手动删除或恢复代码：

1. **恢复 `model_e0.py`**:
   - 删除 `set_systems` 中的 `test_systems` 参数和相关逻辑
   - 删除 `_load_precomputed_graphs` 中的 `is_test` 参数和相关逻辑
   - 删除 `forward` 中的 `is_test` 参数

2. **恢复 `deep_eval.py`**:
   - 删除 `__init__` 中的 `_current_test_sid` 属性
   - 删除 `_eval_model` 中的测试参数传递逻辑，恢复直接调用

3. **恢复 `test.py`**:
   - 删除设置 `test_systems` 的代码块
   - 删除设置 `_current_test_sid` 的代码块

### 方法 3: 使用补丁文件

可以创建一个反向补丁文件来应用回退。

---

## 验证回退

回退后，验证以下功能：

1. ✅ `dp --pt train` 仍然正常工作（training 和 validation）
2. ✅ `dp --pt test` 仍然可以运行（但会回退到动态构建图）
3. ✅ 没有语法错误或导入错误

---

## 注意事项

- 这些改动是**向后兼容**的：如果模型不支持 `is_test` 参数，会自动回退到动态构建图
- `fid` 在测试时假设是顺序的（0, 1, 2, ...），如果测试数据被 shuffle，可能不准确
- 回退后，`dp --pt test` 仍然可以运行，只是不会使用预计算的图

