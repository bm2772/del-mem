---
base_model: Qwen/Qwen3-4B-Instruct-2507
language:
- en
pipeline_tag: text-generation
license: cc-by-4.0
tags:
- delta-mem
- memory-augmented-language-model
- adapter
- qwen3
- long-context
- test-time-memory
- text-generation
---

# δ-mem Qwen3-4B Instruct TSW Adapter

This repository contains the δ-mem TSW adapter for `Qwen/Qwen3-4B-Instruct-2507`, as presented in the paper [δ-mem: Efficient Online Memory for Large Language Models](https://huggingface.co/papers/2605.12357).

δ-mem is a lightweight online memory mechanism that augments a frozen backbone with a compact associative memory state. It projects context into a low-dimensional space and updates a state matrix via delta-rule learning, allowing for efficient long-term memory utilization without full fine-tuning or context extension.

- **Paper:** [https://huggingface.co/papers/2605.12357](https://huggingface.co/papers/2605.12357)
- **Repository:** [https://github.com/declare-lab/delta-Mem](https://github.com/declare-lab/delta-Mem)

## Model Details

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Adapter type | Delta-Mem |
| Variant | TSW, Token State Write |
| Write granularity | `token` |
| Adapter rank | 8 |
| Delta heads | `q`, `o` |
| Training setting | Qasper multi-source memory training, write length 8192 |
| Repository type | Adapter checkpoint, not a merged base model |

## Usage

To use this adapter, you must first install the $\delta$-mem codebase:

```bash
git clone https://github.com/declare-lab/delta-Mem.git
cd delta-Mem
bash scripts/setup_uv_env.sh
```

### Minimal Loading Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core import HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter_dir = "declare-lab/delta-mem_qwen3_4b-instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = HFDeltaMemConfig.from_pretrained(adapter_dir)
attach_delta_mem(model, config)
load_delta_mem_adapter(model, adapter_dir)
model.eval()
```

## Important Notes

This is **not** a standard PEFT LoRA adapter and should not be loaded with `PeftModel` or merged with `merge_and_unload()`. δ-mem requires its specific runtime memory write/read path. To use this adapter, load the frozen base model, attach the modules, and then load the adapter weights using the provided codebase.

## Citation

```bibtex
@misc{lei2026deltamemefficientonlinememory,
      title={$\delta$-mem: Efficient Online Memory for Large Language Models}, 
      author={Jingdi Lei and Di Zhang and Junxian Li and Weida Wang and Kaixuan Fan and Xiang Liu and Qihan Liu and Xiaoteng Ma and Baian Chen and Soujanya Poria},
      year={2026},
      eprint={2605.12357},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.12357}, 
}
```