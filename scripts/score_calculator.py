import json
import os
import sys

# 1. Define the category mapping based on LoCoMo standards
CATEGORY_MAP = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop"
}

# 2. Load the raw results.
#
# Path resolution, in priority order:
#   1. argv[1]                -- score any file:  score_calculator.py path/to/run.jsonl
#   2. $CAIMMS_OUTPUT_DIR     -- set by env.sh, works on every machine
#   3. the original Mahamathi absolute path, as a last-resort fallback
#
# This used to be a bare hardcoded /home/kbasu/... path, which meant the script
# only ran on one machine and silently ignored any filename you passed it.
_DEFAULT_DIR = os.environ.get(
    "CAIMMS_OUTPUT_DIR", "/home/kbasu/arnavbhatt/workmem_test/outputs"
)
file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _DEFAULT_DIR, "workmem_iterret_full.jsonl"
)
if not os.path.exists(file_path):
    sys.exit(
        f"No such results file: {file_path}\n"
        "Pass one explicitly:  python3 scripts/score_calculator.py <run.jsonl>\n"
        "or `source env.sh` first so CAIMMS_OUTPUT_DIR is set."
    )
print(f"scoring: {file_path}\n")

# NOTE: this must be the live ".jsonl" checkpoint (one JSON object per line)
# that pipeline runs append to -- not a sibling ".json" (no "l"), which is a
# stale pre-run-1 snapshot with an outdated schema.
results = []
with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

# 3. Set up our tracking dictionaries
category_stats = {
    "multi_hop": {"total_score": 0.0, "count": 0},
    "temporal": {"total_score": 0.0, "count": 0},
    "open_domain": {"total_score": 0.0, "count": 0},
    "single_hop": {"total_score": 0.0, "count": 0}
}

overall_score = 0.0
total_questions = 0

# 4. Iterate through every single question and tally the scores.
# Category 5 (adversarial) is excluded entirely — not just from the
# per-category breakdown, but from the overall average too.
for item in results:
    cat_id = item.get("category")
    score = item.get("score", 0.0)

    if cat_id == 5:
        continue

    overall_score += score
    total_questions += 1

    if cat_id in CATEGORY_MAP:
        cat_name = CATEGORY_MAP[cat_id]
        category_stats[cat_name]["total_score"] += score
        category_stats[cat_name]["count"] += 1

# 5. Calculate and print the beautiful final report
print(f"\n=========================================")
print(f"      C-AIMMS ITER-RET FINAL SCORES      ")
print(f"=========================================")
print(f"Total Questions Answered: {total_questions}")
print(f"Overall Average Score:    {(overall_score / total_questions):.4f}")
print(f"=========================================\n")

for cat_name, stats in category_stats.items():
    if stats["count"] > 0:
        avg_score = stats["total_score"] / stats["count"]
        print(f"[{cat_name.upper()}]")
        print(f" -> Score: {avg_score:.4f} (from {stats['count']} questions)\n")
