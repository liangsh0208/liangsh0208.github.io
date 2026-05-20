---
created: 2026-05-06
---

# 07_Checkpoint与延迟初始化

## 1. 目标论证

**为什么需要延迟初始化？**

大模型运行OOM可能发生在模型创建阶段：
- Llama-70B参数: 140GB (fp16)
- TP=8时每个rank仍需加载140GB才能切分
- 延迟初始化避免"先膨胀再切分"的内存峰值

**Lazy Loading优势：**
- 先在meta device创建骨架
- TP/PP/CP切分骨架(几乎零内存)
- 实际加载权重时再分配内存

## 2. 工作假设

**硬件假设：**
- 有足够的CPU RAM存储完整检查点
- 各rank可以独立访问存储

**存储格式：**
- 支持safetensors格式
- 支持PyTorch原生.pth格式
- 多维度分片存储(TP × PP)

## 3. 解决方案

### 3.1 延迟初始化上下文

```python
# checkpoint.py
@contextlib.contextmanager
def init_model_with_dematerialized_weights():
    """
    在meta device上创建模型骨架，不分配实际内存
    """
    # 保存原始的register_parameter方法
    old_register_parameter = nn.Module.register_parameter
    
    def register_empty_parameter(module, name, param):
        """将参数替换为meta device上的空张量"""
        old_register_parameter(module, name, param)
        
        if param is not None:
            # 关键：将参数移到meta device
            module._parameters[name] = nn.Parameter(
                param.to(torch.device("meta")),
                requires_grad=param.requires_grad
            )
    
    # 替换方法
    nn.Module.register_parameter = register_empty_parameter
    
    try:
        yield
    finally:
        # 恢复原始方法
        nn.Module.register_parameter = old_register_parameter


def init_model_with_materialized_weights(model, config, device, checkpoint_path=None):
    """
    为meta device上的模型分配实际内存并加载权重
    """
    pgm = get_process_group_manager()
    
    # 遍历所有meta参数
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            # 创建实际张量
            materialized = torch.empty(
                param.shape,
                dtype=param.dtype,
                device=device
            )
            
            # 初始化或加载权重
            if checkpoint_path:
                checkpoint = torch.load(get_shard_path(checkpoint_path, pgm), map_location="cpu")
                if name in checkpoint:
                    materialized.copy_(checkpoint[name])
                else:
                    # 新层初始化
                    init_weights(materialized, name)
            else:
                # 随机初始化
                init_weights(materialized, name)
            
            # 替换meta参数
            parent_name, param_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # 删除meta参数，设置新的materialized参数
            delattr(parent, param_name)
            setattr(parent, param_name, nn.Parameter(materialized, requires_grad=param.requires_grad))
    
    return model
```

### 3.2 使用流程

```python
def create_model(config, device):
    """完整的延迟初始化流程"""
    
    # Step 1: 在meta device创建骨架(几乎零内存)
    with init_model_with_dematerialized_weights():
        model = LlamaForCausalLM(config)
        
        # Step 2: 应用TP切分(切分meta参数，无额外内存)
        if config.use_tp:
            model = apply_tensor_parallel(model)
        
        # Step 3: 应用PP切分(选择对应层)
        if config.use_pp:
            model = PipelineParallel(model, config)
    
    # Step 4: 分配实际内存并加载/初始化权重
    model = init_model_with_materialized_weights(
        model, config, device, 
        checkpoint_path=config.checkpoint_path
    )
    
    # Step 5: 应用CP(无需修改参数)
    if config.use_cp:
        model = apply_context_parallel(model)
    
    # Step 6: 应用DP
    if config.use_dp:
        model = DataParallelBucket(model)
    
    return model
```

### 3.3 CheckpointManager

```python
class CheckpointManager:
    """
    支持多维度分片的检查点管理
    每个TP × PP组合保存独立文件
    """
    def save(self, model, optimizer, step, out_dir):
        pgm = get_process_group_manager()
        
        # 非TP/DP进程只需保存部分参数
        if not self._should_save_rank(pgm):
            return
        
        state_dict = {
            # 模型参数(已按TP/PP切分)
            "model": {k: v.clone() for k, v in model.named_parameters()},
            # 优化器状态
            "optimizer": optimizer.state_dict(),
            # 其他训练状态
            "step": step,
            "global_step": step,
        }
        
        checkpoint_path = self._get_checkpoint_path(out_dir, pgm)
        
        if config.use_safetensors:
            # 使用safetensors格式
            from safetensors.torch import save_file
            save_file(state_dict["model"], checkpoint_path)
        else:
            # PyTorch格式
            torch.save(state_dict, checkpoint_path)
    
    def _get_checkpoint_path(self, out_dir, pgm):
        """生成包含维度信息的文件名"""
        tp_rank = pgm.tp_rank if pgm else 0
        tp_world = pgm.tp_world_size if pgm else 1
        pp_rank = pgm.pp_rank if pgm else 0
        pp_world = pgm.pp_world_size if pgm else 1
        
        return os.path.join(
            out_dir,
            f"weights_"
            f"tp_rank_{tp_rank}_{tp_world}_"
            f"pp_rank_{pp_rank}_{pp_world}.pth"
        )
    
    def load(self, model, checkpoint_dir):
        """加载对应维度的检查点"""
        pgm = get_process_group_manager()
        checkpoint_path = self._get_checkpoint_path(checkpoint_dir, pgm)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)
        
        return state_dict.get("step", 0)
```

### 3.4 从HuggingFace加载

```python
def load_from_pretrained(model, model_name, device):
    """
    从HuggingFace checkpoint加载，支持分片到TP/PP
    """
    from transformers import AutoModelForCausalLM
    
    # 加载原始模型(可能大显存占用)
    pretrained = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"  # 先加载到CPU
    )
    
    # 提取需要加载的参数名
    pgm = get_process_group_manager()
    target_names = [n for n, _ in model.named_parameters()]
    
    # 分TP/PP加载
    for target_name in target_names:
        # 映射picotron层名到HuggingFace层名
        source_name = _map_layer_name(target_name, pgm)
        
        if source_name in pretrained.state_dict():
            param = pretrained.state_dict()[source_name]
            
            # 根据TP/PP位置切分参数
            if is_qkv_proj(target_name):
                param = _shard_for_tp(param, pgm.tp_rank, pgm.tp_world_size)
            elif is_pp_layer(target_name):
                if should_skip_layer(target_name, pgm):
                    continue  # 当前PP rank不持有该层
            
            with torch.no_grad():
                model.get_parameter(target_name).copy_(param)
    
    del pretrained  # 释放大模型内存
    torch.cuda.empty_cache()
```

## 4. 故障模式

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| `RuntimeError: meta tensors` | 有参数未materialized | 检查所有参数都经过init_model_with_materialized_weights |
| 加载权重形状不匹配 | TP/PP切分后参数名映射错误 | 检查_map_layer_name函数 |
| Checkpoint文件缺失 | PP/TP rank映射错误 | 确认文件名中的rank与实际PGM一致 |
| 恢复后损失跳变 | 优化器状态未保存/加载 | 确保optimizer.state_dict()包含在checkpoint中 |
| 内存峰值仍然存在 | meta device创建前有参数分配 | 确保在上下文管理器内完成所有nn.Module创建 |

### 调试技巧

```python
# 检查meta参数
for name, param in model.named_parameters():
    if param.device.type == "meta":
        print(f"Warning: {name} is still on meta device")

# 统计各rank的检查点大小
checkpoint_size = sum(
    p.numel() * p.element_size() 
    for p in model.parameters()
) / 1e9
print(f"Rank {pgm.global_rank}: {checkpoint_size:.2f}GB parameters")
```

---
*上一页: [06_模型实现.md](06_模型实现.md) | 下一页: [08_训练脚本解析.md](08_训练脚本解析.md)*
