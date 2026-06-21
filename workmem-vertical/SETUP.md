# WORKMEM vertical setup

1. Clone with submodules:
   git clone --recurse-submodules <C-AIMMS-url>
   (or if already cloned: git submodule update --init --recursive)

2. Download the model and adapter:
   huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
   huggingface-cli download declare-lab/delta-mem_qwen3_4b-instruct --local-dir models/delta-mem_qwen3_4b-instruct

3. Build the environment:
   cd delta-Mem
   uv venv .venv --python 3.11
   source .venv/bin/activate
   uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   uv pip install -r requirements.txt

4. Run the smoke test (from the delta-Mem directory, with workmem-vertical's files
   copied or symlinked into deltamem/workmem/):
   PYTHONPATH=. python deltamem/workmem/test_osam_smoke.py

This vertical also requires a small patch to delta-Mem/deltamem/core/delta_impl.py
(already applied on the workmem-osam branch this submodule points to):

def set_delta_mem_write_granularity(model, granularity):
    granularity = normalize_memory_write_granularity(granularity)
    for _, module in iter_delta_mem_modules(model):
        module.memory_write_granularity = granularity
