from __future__ import annotations

import argparse
import json
import os

from iterret.ctc_graph import CueTagContentGraph
from iterret.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from iterret.memory_builder import DEFAULT_MAX_CHARS_PER_CALL, DialogueTurn, build_ctc_graph_from_dialogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ITERRET CTC memory graph offline.")
    parser.add_argument("--use-real-llm", action="store_true",
                         help="Use a local OpenAI-compatible LLM server instead of the mock.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dialogue", default=None,
                         help="Path to a JSON file with [{speaker, text, time}, ...] turns. "
                              "Defaults to the bundled sample dialogue.")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--max-chars-per-call", type=int, default=DEFAULT_MAX_CHARS_PER_CALL,
                         help="Cap on episode-text characters packed into a single semantic/topic "
                              "extraction LLM call (default %(default)s; lower this for smaller "
                              "context-window servers, e.g. a -2507-tagged Qwen3 server run with "
                              "--max-model-len 8192).")
    return parser.parse_args()


def _load_turns(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        raw_turns = json.load(fh)
    return [DialogueTurn(speaker=t.get("speaker", "Unknown"), text=t.get("text", ""),
                          time=t.get("time")) for t in raw_turns]


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    graph_path = os.path.join(args.data_dir, "ctc_graph.json")

    turns = _load_turns(args.dialogue) if args.dialogue else sample_dialogue_turns()
    print(f"[memory-builder] distilling {len(turns)} dialogue turn(s) into a CTC graph...")

    llm = (OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
           if args.use_real_llm else MockLLMClient())
    if args.use_real_llm:
        print(f"[memory-builder] using real LLM at {llm.base_url} (model={llm.model})")

    graph = build_ctc_graph_from_dialogue(turns, llm, max_chars_per_call=args.max_chars_per_call)
    graph.save(graph_path)

    n_cues = len(graph.cues)
    n_episodic = sum(1 for c in graph.contents.values() if c.layer == "episodic")
    n_semantic = sum(1 for c in graph.contents.values() if c.layer == "semantic")
    n_topic = sum(1 for c in graph.contents.values() if c.layer == "topic")
    n_links = sum(len(v) for v in graph.cue_tag_to_content.values())
    print(f"[memory-builder] saved graph to {graph_path}: {n_cues} cue(s), "
          f"{n_episodic} episodic + {n_semantic} semantic + {n_topic} topic content node(s), "
          f"{n_links} cue-tag-content link(s)")


if __name__ == "__main__":
    main()
