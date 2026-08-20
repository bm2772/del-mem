import json

# 1. Define the category mapping based on LoCoMo standards
CATEGORY_MAP = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop"
}

# 2. Load the raw results from your successful run.
# This is the live checkpoint file (one JSON object per line) that every
# pipeline run actually appends to -- the sibling ".json" (no "l") file is
# a stale, pre-run-1 snapshot with an outdated schema. Don't point back at it.
file_path = "/home/kbasu/arnavbhatt/workmem_test/outputs/workmem_iterret_full.jsonl"
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
