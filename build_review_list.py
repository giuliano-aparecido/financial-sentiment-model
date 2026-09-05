"""Builds human-readable, ticker-sorted review files - one for the
synthetic dataset, one for the real dataset - each with every row for the
same ticker grouped together (train and val both included, labeled), so
manual review can look at one company at a time instead of jumping around
a randomly-ordered file. Read-only: doesn't touch the source JSONL files.

Usage:
    python build_review_list.py

Writes review_synthetic_dataset_by_ticker.txt and
review_real_dataset_by_ticker.txt in the current directory.
"""

import json

DATASETS = [
    {
        "label": "synthetic",
        "sources": [("train", "dataset_train.jsonl"), ("val", "dataset_val.jsonl")],
        "output_file": "review_synthetic_dataset_by_ticker.txt",
    },
    {
        "label": "real",
        "sources": [("train", "dataset_train_real.jsonl"), ("val", "dataset_val_real.jsonl")],
        "output_file": "review_real_dataset_by_ticker.txt",
    },
]


def build_review(label, sources, output_file):
    rows = []
    for split, filename in sources:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)

    rows.sort(key=lambda r: (r["ticker"], r["_split"]))

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"{label.upper()} dataset review list: {len(rows)} total rows "
                   f"({sum(1 for r in rows if r['_split']=='train')} train, "
                   f"{sum(1 for r in rows if r['_split']=='val')} val), grouped by ticker.\n")
        out.write("=" * 100 + "\n\n")

        current_ticker = None
        for row in rows:
            if row["ticker"] != current_ticker:
                current_ticker = row["ticker"]
                out.write(f"\n{'#' * 100}\n# TICKER: {current_ticker}\n{'#' * 100}\n\n")

            output = json.loads(row["output"])
            task = row.get("task")

            out.write(f"--- [{row['_split']}] {row['ticker']} ({task}) ---\n")
            if task == "reaction":
                # Task A rows only carry the news block and the classified
                # reaction - no market_data/valuation/earnings/recommendation,
                # see generate_synthetic_dataset.py/generate_real_dataset.py's
                # own task_a_row builders.
                out.write(f"News:\n{row['news']}\n\n")
                out.write(f"News reaction (dataset label): {output['news_reaction']}\n")
            elif task == "analysis":
                out.write(f"User Question: {row['user_query']}\n\n")
                out.write(f"Market Data:\n{row['market_data']}\n\n")
                out.write(f"Valuation:\n{row['valuation']}\n\n")
                out.write(f"Earnings:\n{row['earnings']}\n\n")
                out.write(f"News:\n{row['news']}\n\n")
                out.write(f"Recommendation (dataset label): {row.get('recommendation', '')}\n")
                out.write(f"Reasoning (dataset label): {output['reasoning']}\n")
                out.write(f"Answer (dataset label): {output['answer']}\n")
            else:
                raise ValueError(f"Unknown task: {task!r}")
            out.write("\n" + "-" * 100 + "\n\n")

    print(f"Wrote {output_file}: {len(rows)} rows, {len(set(r['ticker'] for r in rows))} distinct tickers.")


def main():
    for dataset in DATASETS:
        build_review(dataset["label"], dataset["sources"], dataset["output_file"])


if __name__ == "__main__":
    main()
