# C-AIMMS Boundary Creator Documentation

The `caimms_boundary_creator.py` module serves as the primary **Perception and Segmentation Interface** for the C-AIMMS (Cognitive Artificial Intelligence Memory Management System) architecture. 

It is designed to wrap around any causal Language Model (like Qwen, Llama) and intercept its generation process to emit **absolute token indices** marking the boundaries of new semantic episodes. 

## Architectural Role in C-AIMMS

Unlike the original EM-LLM architecture which tightly coupled boundary detection with internal GPU episodic memory storage, the `CAIMMSBoundaryEmitter` is **strictly a lightweight index emitter**. 

It adheres to the C-AIMMS philosophy of separating *Perception* from *Storage*:
1. **Perception (This Module):** Uses Bayesian Surprise and Graph Modularity to determine *when* a topic or context shift has occurred in a continuous token stream.
2. **Storage (HETREP):** It does NOT store the KV-cache of the episodes itself. Instead, it emits the `[start_idx, end_idx]` boundaries so an external orchestrator can slice the text and encode it into the HETREP (Hypergraph Event-Temporal Representation) Vector Database.

## Core Components

### 1. The Wrapper (`CAIMMSBoundaryEmitter`)
A PyTorch `nn.Module` that encapsulates the HuggingFace base model. 
- It intercepts `outputs.logits` and `outputs.past_key_values` during the standard `forward()` pass.
- It slices the `past_key_values` tensor to extract only the Key embeddings of the **Uncommitted Working Memory** (tokens that haven't been sealed into an episode yet).
- It passes these to the `StatefulSurpriseBoundary` tracker.

### 2. The 2-Stage Boundary Logic
The boundary creation process happens in two distinct mathematical stages, executed by `em_llm.attention.boundary_creator`:

#### Stage 1: Bayesian Surprise (Spike Detection)
As the model predicts the next token, the pipeline calculates the "Surprise" (Cross-Entropy Loss/Negative Log-Likelihood) of each token.
If the surprise of a token exceeds a rolling historical threshold ($> \gamma \times \text{std} + \text{mean}$), it is flagged as a potential boundary.
*This stage ensures boundaries are triggered by genuine semantic shifts (e.g., speaker changes, new topics).*

#### Stage 2: Graph Modularity (Temporal Refinement)
To prevent creating boundaries in the middle of a cohesive thought, the pipeline takes the Key-embeddings of the uncommitted tokens and computes an Adjacency Matrix (`Stacked A`) using Cosine Similarity. 
It treats the working memory as a temporal graph and slightly shifts the raw surprise triggers to the nearest local minima/maxima of graph modularity. 

## Key Hyperparameters

When initializing `CAIMMSBoundaryEmitter`, the following parameters dictate the sensitivity and size of your episodes:

- **`model_name_or_path`** (default: `"Qwen/Qwen3-4B-Instruct"`): The base causal language model to wrap.
- **`surprisal_threshold_gamma`** (default: `1.5`): The statistical strictness of boundary creation. 
  - *Lower (e.g., 1.1)*: Highly sensitive. Will trigger boundaries on minor topic shifts. Can result in many small, uniform episodes.
  - *Higher (e.g., 1.5 - 2.0)*: Stricter. Will only trigger boundaries on major contextual shifts, allowing episodes to grow larger and more varied.
- **`min_block_size`** (default: `8`): The absolute minimum number of tokens an episode can contain.
- **`n_local`** & **`n_init`**: Controls the size of the rolling window used to calculate the mean and standard deviation of historical surprise.

## Memory Bounding (OOM Prevention)

To prevent CUDA Out Of Memory errors on infinitely long continuous streams, the wrapper actively manages the active Working Memory. 
When a boundary is finalized, the wrapper **mathematically slices** the `past_key_values` tensor to physically discard the committed tokens from the GPU's KV cache. The KV cache is strictly bounded to only the current uncommitted tokens, ensuring the GPU memory profile remains flat ($O(1)$) regardless of how long the conversation runs!

## Integration Example

```python
from caimms_boundary_creator import CAIMMSBoundaryEmitter
import torch

# 1. Initialize the Emitter
emitter = CAIMMSBoundaryEmitter(
    model_name_or_path="Qwen/Qwen3-4B-Instruct",
    surprisal_threshold_gamma=1.5
)

input_ids = tokenizer("...massive continuous context...", return_tensors="pt").input_ids

# 2. Process in chunks
chunk_size = 2048
past_key_values = None

for i in range(0, input_ids.shape[1], chunk_size):
    chunk = input_ids[:, i:i+chunk_size]
    
    # The forward pass automatically calculates boundaries and emits their absolute indices!
    # It also dynamically slices past_key_values to prevent OOM.
    outputs = emitter(input_ids=chunk, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
```
