import pathlib
import datasets
import fire
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def sample_per_score_and_write_tsvs(input_path: str = "./labelled/",
                                   output_path: str = "./annotation/",
                                   out_prefix: str = "sampled",
                                   per_score: int = 24,
                                   max_length: int = 1500,
                                   chunk_size: int = 20,
                                   seed: int | None = 42) -> list:
    """Load dataset, sample up to `per_score` items for each llm_score in [0..5],
    combine, shuffle, add empty column and write chunked TSV files.

    Args:
        input_path: Path for datasets.load_dataset
        output_path:
        out_prefix: prefix used to write TSV files; files will be named {out_prefix}_part{n}.tsv
        per_score: how many items to pick per score value (0..5)
        max_length: maximum allowed length for `text` and `wiki_text` fields (characters)
        chunk_size: number of rows per output TSV file (default 20)
        seed: seed used to shuffle subset selection and final shuffling

    Returns:
        The list of combined selected entries (list of dicts)
    """
    ds = datasets.load_dataset(input_path, split="train")

    def _to_int_score(val):
        try:
            return int(val)
        except Exception:
            try:
                return int(float(val))
            except Exception:
                return None

    def _to_float_score(val):
        """Safely convert a score value to float, return None on failure."""
        try:
            return float(val)
        except Exception:
            return None

    all_selected = []
    for score in range(6):
        # Build candidate list for this llm_score with computed diff between wiki_score and text_score
        candidates = []
        for row in ds:
            sc = _to_int_score(row.get("llm_score"))
            if sc != score:
                continue
            text = (row.get("text") or "")
            wiki_text = (row.get("wiki_text") or "")
            # numeric scores that determine difference
            wiki_score = _to_float_score(row.get("wiki_score"))
            text_score = _to_float_score(row.get("text_score"))
            if wiki_score is None or text_score is None:
                diff = -1.0
            else:
                diff = abs(wiki_score - text_score)
            candidates.append({
                "row": row,
                "diff": diff,
                "text": text,
                "wiki_text": wiki_text,
                "wiki_score": wiki_score,
                "text_score": text_score,
            })

        # sort descending by difference
        candidates.sort(key=lambda r: r["diff"], reverse=True)

        # compute candidate diffs (exclude invalid diff values < 0)
        candidate_diffs = [c["diff"] for c in candidates if c["diff"] >= 0]
        def _stats(vals: list[float]):
            if not vals:
                return (None, None, None)
            mn = min(vals)
            mx = max(vals)
            avg = sum(vals) / len(vals)
            return (mn, mx, avg)
        c_mn, c_mx, c_avg = _stats(candidate_diffs)
        print(f"score={score}\tcandidates:\tcount={len(candidates)}\tdiffs_count={len(candidate_diffs)}\tmin={c_mn}\tmax={c_mx}\tavg={c_avg}")

        selected = []
        selected_ids = set()
        # first choose items that already satisfy length constraints
        for c in candidates:
            if len(selected) >= per_score:
                break
            if len(c["text"]) <= max_length and len(c["wiki_text"]) <= max_length:
                row = c["row"]
                r_id = row.get("id")
                if r_id in selected_ids:
                    continue
                selected.append({
                    "text": c["text"],
                    "wiki_text": c["wiki_text"],
                    "id": row.get("id"),
                    "llm_score": _to_int_score(row.get("llm_score")),
                    "diff": c["diff"],
                    "wiki_score": c["wiki_score"],
                    "text_score": c["text_score"],
                })
                selected_ids.add(r_id)

        # If not enough, add more candidates by difference order, truncating text fields to max_length
        if len(selected) < per_score:
            for c in candidates:
                if len(selected) >= per_score:
                    break
                row = c["row"]
                text = c["text"]
                wiki_text = c["wiki_text"]
                r_id = row.get("id")
                if r_id in selected_ids:
                    continue
                if len(text) <= max_length and len(wiki_text) <= max_length:
                    # already selected or was a good candidate; ensure not duplicated
                    continue
                # truncate
                if len(text) > max_length:
                    text = text[:max_length]
                if len(wiki_text) > max_length:
                    wiki_text = wiki_text[:max_length]
                selected.append({
                    "text": text,
                    "wiki_text": wiki_text,
                    "id": row.get("id"),
                    "llm_score": _to_int_score(row.get("llm_score")),
                    "diff": c["diff"],
                    "wiki_score": c["wiki_score"],
                    "text_score": c["text_score"],
                })
                selected_ids.add(r_id)

        n = len(selected)
        # compute stats for selected diffs
        selected_diffs = [s.get("diff") for s in selected if s.get("diff") is not None and s.get("diff") >= 0]
        s_mn, s_mx, s_avg = _stats(selected_diffs)
        print(f"score={score}\tselected:\tcount={n}\tdiffs_count={len(selected_diffs)}\tmin={s_mn}\tmax={s_mx}\tavg={s_avg}")
        all_selected.extend(selected)
        # summary already printed above

    # Shuffle combined
    import random
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_selected)

    # Prepare DataFrame and chunk to files
    columns = ["educational score 0-5", "text", "wiki_text", "id", "llm_score", "wiki_score", "text_score"]
    # Fill educational with empty strings
    for item in all_selected:
        item["educational score 0-5"] = ""
        # Ensure string conversion for text fields
        if item.get("text") is None:
            item["text"] = ""
        if item.get("wiki_text") is None:
            item["wiki_text"] = ""
        if item.get("id") is None:
            item["id"] = ""
        if item.get("llm_score") is None:
            item["llm_score"] = ""
        if item.get("wiki_score") is None:
            item["wiki_score"] = ""
        if item.get("text_score") is None:
            item["text_score"] = ""

    df = pd.DataFrame(all_selected)
    # Reorder columns
    df = df[[c for c in columns]]

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    import math
    total = len(df)
    if total == 0:
        print("No samples selected; nothing to write.")
        return all_selected
    n_files = math.ceil(total / chunk_size)
    for i in range(n_files):
        start = i * chunk_size
        end = min(total, (i + 1) * chunk_size)
        sub = df.iloc[start:end]
        out_file = output_path / f"{out_prefix}_{i}.tsv"
        sub.to_csv(out_file, sep="\t", index=False)
        print(f"Wrote {len(sub)} rows to {out_file}")

    return all_selected

def main(input_path: str = "./labelled/",
         extract: bool = False,
         extract_out: str | None = None,
         sample_out_prefix: str | None = "batch",
         per_score: int = 24,
         max_length: int = 1500,
         chunk_size: int = 20,
         seed: int | None = 42):

    combined = sample_per_score_and_write_tsvs(input_path=input_path,
                                                output_path=extract_out or "./annotation/",
                                                out_prefix=sample_out_prefix or "sampled",
                                                per_score=per_score,
                                                max_length=max_length,
                                                chunk_size=chunk_size,
                                                seed=seed)
    print(f"Selected {len(combined)} combined samples")


if __name__ == '__main__':
    fire.Fire(main)
