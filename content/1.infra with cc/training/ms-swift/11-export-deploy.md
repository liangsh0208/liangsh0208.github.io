# ms-swift 导出量化与部署

模型训练完成后，ms-swift 提供完整的后处理流程：LoRA 合并、量化导出、格式转换（Ollama），以及数据集缓存等辅助功能。

---

## 1. SwiftExport — 导出管道

**文件**: `swift/pipelines/export/` 目录

```python
class SwiftExport(SwiftPipeline):
    def run(self):
        if self.args.merge_lora:
            merge_lora(self.args)
        if self.args.quant_bits:
            quantize_model(self.args)
        if self.args.to_ollama:
            export_to_ollama(self.args)
```

---

## 2. LoRA Merge

### 2.1 合并流程

**文件**: `swift/pipelines/export/merge_lora.py`

```python
def merge_lora(args, device_map='cpu'):
    """将 LoRA adapter 权重合并回基模型"""
    # 1. 加载基模型
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map)
    
    # 2. 加载 LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapters[0])
    
    # 3. 合并权重
    model = model.merge_and_unload()
    
    # 4. 保存合并后的模型
    model.save_pretrained(args.output_dir)
    # 复制 tokenizer 和 config
    shutil.copy(os.path.join(args.model, 'tokenizer.json'), args.output_dir)
```

### 2.2 合并公式

对于每个应用了 LoRA 的线性层 `W`：

```
W_merged = W_base + (lora_B @ lora_A) * scaling
where scaling = lora_alpha / lora_rank
```

合并后的模型是完全标准的 transformers 模型，可直接用于推理/量化/部署。

### 2.3 CLI 使用

```bash
swift merge-lora \
    --model Qwen/Qwen3-4B-Instruct \
    --adapters output/lora-checkpoint \
    --output_dir output/merged-model
```

---

## 3. 量化导出

### 3.1 GPTQ

**文件**: `swift/pipelines/export/quant.py`

```python
def quantize_gptq(model_path, output_path, bits=4, group_size=128):
    """GPTQ 4-bit 量化"""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, quantize_config)
    model.quantize(calib_dataset)
    model.save_quantized(output_path)
```

### 3.2 AWQ

```python
def quantize_awq(model_path, output_path, bits=4):
    """AWQ 量化"""
    from awq import AutoAWQForCausalLM
    
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    model.quantize(calib_dataset)
    model.save_quantized(output_path)
```

### 3.3 BNB (BitsAndBytes)

训练时即可指定量化精度：
```bash
swift sft --model xxx --quant_bits 4  # QLoRA 训练
```

### 3.4 FP8

NVIDIA Hopper 架构支持 FP8 量化：
```bash
swift export --model xxx --quant_method fp8 --output_dir output/fp8
```

### 3.5 量化矩阵

| 量化方法 | 位数 | 训练支持 | 推理后端 | 文件 |
|---------|-----|---------|---------|------|
| BNB | 4/8-bit | 是 (QLoRA) | transformers/vLLM/LMDeploy | 推理时自动检测 |
| GPTQ | 4-bit | 否 | vLLM/LMDeploy | `swift/pipelines/export/quant.py` |
| AWQ | 4-bit | 否 | vLLM/LMDeploy | `swift/pipelines/export/quant.py` |
| AQLM | 2/4-bit | 否 | transformers | 推理时自动检测 |
| FP8 | 8-bit (E4M3) | 否 | transformers+vLLM | Hopper 架构 |
| HQQ | 4-bit | 否 | transformers | 推理时自动检测 |

---

## 4. Ollama 导出

**文件**: `swift/pipelines/export/ollama.py`

将模型导出为 Ollama 可导入的格式：

```python
def export_to_ollama(args):
    """导出为 Ollama Modelfile 格式"""
    # 1. 生成 Modelfile
    modelfile = f'''
FROM {os.path.abspath(args.model)}
TEMPLATE """{template.template_str}"""
PARAMETER temperature 0.7
PARAMETER stop <|im_end|>
'''
    
    # 2. 保存 Modelfile
    with open(os.path.join(args.output_dir, 'Modelfile'), 'w') as f:
        f.write(modelfile)
    
    # 3. 用户可执行: ollama create my-model -f Modelfile
```

---

## 5. Dataset Caching

**文件**: `swift/pipelines/export/cached_dataset.py`

```python
def cache_dataset(args):
    """预处理和缓存数据集到磁盘"""
    dataset = load_dataset(args.dataset)
    
    # 预编码
    template = get_template(args.template)
    encoded_dataset = dataset.map(template.encode, num_proc=args.num_proc)
    
    # 保存到 disk
    encoded_dataset.save_to_disk(args.output_dir)
```

用途：
- 大规模数据集的预处理加速（避免每次训练重复编码）
- 分布式训练中数据集的一致性保障

---

## 6. 部署服务详见

推理部署功能已移至 [`inference-system.md`](09-inference-system.md)。

---

## 7. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| 导出管道 | `swift/pipelines/export/` 目录 |
| LoRA 合并 | `swift/pipelines/export/merge_lora.py::merge_lora()` |
| 量化引擎 | `swift/pipelines/export/quant.py::QuantEngine` |
| GPTQ 量化 | `swift/pipelines/export/quant.py` |
| AWQ 量化 | `swift/pipelines/export/quant.py` |
| Ollama 导出 | `swift/pipelines/export/ollama.py::export_to_ollama()` |
| 数据集缓存 | `swift/pipelines/export/cached_dataset.py` |
| Merge CLI | `swift/cli/merge_lora.py` |
| Export CLI | `swift/cli/export.py` |
