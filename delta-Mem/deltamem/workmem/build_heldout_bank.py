"""Ablation C3: bootstrap an IterRet experience bank on HELD-OUT LoCoMo
conversations, for use with ITERRET_BANK_PATH on LoCoMo-4 (conversations 1-4)
and LongMemEval.

An earlier bank built from one conversation hurt (multi-hop -0.022, temporal
-0.024); this one draws on several conversations that are never evaluated in
LoCoMo-4. Runs under run_pipeline.sh (it needs the vLLM server):

  EVAL_MODULE=deltamem.workmem.build_heldout_bank \
  WORKMEM_OUTPUT_FILE=$CAIMMS_OUTPUT_DIR/experience_bank_heldout.json \
      bash scripts/run_pipeline.sh

  C3_BANK_SAMPLES   conversation indices to bootstrap on (default "4 5 6 7 8 9")
  C3_BANK_QUESTIONS seed questions per conversation (default 15; categories 1-4,
                    adversarial category 5 left out, evenly spaced through the list)

Graphs come from $CAIMMS_OUTPUT_DIR/graph_cache/sample_<i>.json (built by the
LoCoMo eval, e.g. B1); missing ones are built and cached the same way.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import build_default_embedding_backend
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue
from iterret.offline_pipeline import collect_trajectories, construct_experience_banks

from deltamem.workmem.eval_locomo_iterret_mock import DATA_FILE, VLLM_BASE_URL, VLLM_MODEL_NAME
from deltamem.workmem.longctx_retrieval import ITERRET_MAX_ITERATIONS

OUT = os.environ["WORKMEM_OUTPUT_FILE"]
SAMPLES = [int(x) for x in os.environ.get("C3_BANK_SAMPLES", "4 5 6 7 8 9").split()]
PER_CONV = int(os.environ.get("C3_BANK_QUESTIONS", "15"))
GRAPH_DIR = Path(os.environ.get("CAIMMS_OUTPUT_DIR", Path(OUT).parent)) / "graph_cache"


def turns_of(sample: dict) -> list:
    conv = sample.get("conversation", {})
    nums = sorted(int(k.split("_")[1]) for k in conv if k.startswith("session_") and k.split("_")[1].isdigit()
                  and not k.endswith("date_time"))
    return [DialogueTurn(speaker=d.get("speaker", "Unknown"), text=d.get("text", ""),
                         time=str(conv.get(f"session_{n}_date_time", f"Session {n}")))
            for n in nums for d in conv.get(f"session_{n}", [])]


def seed_questions(sample: dict) -> list:
    qs = [q["question"] for q in sample.get("qa", []) if q.get("category") in (1, 2, 3, 4)]
    if len(qs) <= PER_CONV:
        return qs
    step = len(qs) / PER_CONV
    return [qs[int(i * step)] for i in range(PER_CONV)]


def main() -> None:
    samples = json.load(open(DATA_FILE))
    llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)
    backend = build_default_embedding_backend()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i in SAMPLES:
        path = GRAPH_DIR / f"sample_{i}.json"
        if path.exists():
            graph = CueTagContentGraph.load(str(path))
        else:
            print(f"[c3] sample {i}: building graph (not cached)...", flush=True)
            graph = build_ctc_graph_from_dialogue(turns_of(samples[i]), llm)
            graph.save(str(path))
        graph.attach_embedder(backend)
        qs = seed_questions(samples[i])
        print(f"[c3] sample {i}: {len(qs)} bootstrap trajectories...", flush=True)
        records += collect_trajectories(qs, graph, llm, max_iterations=ITERRET_MAX_ITERATIONS)
    print(f"[c3] scoring {len(records)} trajectories and distilling experience...", flush=True)
    bank = construct_experience_banks(records, llm)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    bank.save(OUT)
    print(f"[c3] saved {OUT}: {len(bank.planning_bank)} Planning + {len(bank.reflection_bank)} Reflection "
          f"entries from samples {SAMPLES}", flush=True)


if __name__ == "__main__":
    main()
