from __future__ import annotations

"""
Build HFT (Hybrid Fine-Tuning) dataset for ARR-LHRM.

We construct a supervised dataset that teaches the model to follow 3 modes:
- direct: answer directly (<no_think>)
- retrieve: answer with provided evidence (<no_think> + context)
- reason: answer with explicit reasoning pattern (<think> ... </think>)

Output format matches `train/train.py` expected schema:
  [
    [ [prompt_part, answer_part], [False, True] ],
    ...
  ]
"""

import argparse
import json
import random
from typing import Any, Dict, List


def build_prompt(query: str, mode: str, title: str = "", context: str = "") -> str:
    header = f"Question: {query}\n\n"
    if mode == "direct":
        return header
    if mode == "retrieve":
        return header
    if mode == "reason":
        return header
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_path", type=str, required=True, help="jsonl file, each line a QA instance")
    ap.add_argument("--doc_path", type=str, required=True, help="docs.json (same as repo)")
    ap.add_argument("--output_path", type=str, required=True, help="output json list for train/train.py")
    ap.add_argument(
        "--router_labels_output",
        type=str,
        default="",
        help="Optional: output jsonl with {query, mode} for training router HFT.",
    )
    ap.add_argument("--seed", type=int, default=42)

    # Sampling ratios
    ap.add_argument("--direct_ratio", type=float, default=0.34)
    ap.add_argument("--retrieve_ratio", type=float, default=0.33)
    ap.add_argument("--reason_ratio", type=float, default=0.33)
    ap.add_argument("--max_instances", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    random.seed(args.seed)

    docs = json.load(open(args.doc_path, encoding="utf-8"))
    doc_dict: Dict[str, Dict[str, Any]] = {str(d["wikipedia_id"]): d for d in docs}

    qa: List[Dict[str, Any]] = []
    with open(args.qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa.append(json.loads(line))
    if args.max_instances and args.max_instances > 0:
        qa = qa[: args.max_instances]

    ratios = [args.direct_ratio, args.retrieve_ratio, args.reason_ratio]
    s = sum(ratios)
    ratios = [r / s for r in ratios]

    out: List[List[Any]] = []
    router_labels: List[Dict[str, str]] = []
    for inst in qa:
        query = inst.get("query", "")
        answers = inst.get("answers", [])
        if not query or not answers:
            continue

        # pick first answer as supervision (doc format varies across datasets)
        ans0 = answers[0]
        answer_text = (ans0.get("answer", "") or "").strip()
        if not answer_text:
            # skip empty answers for HFT
            continue

        # retrieve context: pick first context if present
        title = ""
        context = ""
        contexts = ans0.get("context", [])
        if contexts:
            c0 = contexts[0]
            did = str(c0.get("wikipedia_id", ""))
            cid = int(c0.get("chunk_id", 0) or 0)
            if did in doc_dict:
                title = str(doc_dict[did]["title"])
                chunks = doc_dict[did].get("chunks", [])
                if 0 <= cid < len(chunks):
                    context = str(chunks[cid])

        mode = random.choices(["direct", "retrieve", "reason"], weights=ratios, k=1)[0]
        if mode == "retrieve" and (not title or not context):
            # if we can't construct evidence, fall back
            mode = "direct"

        prompt = build_prompt(query=query, mode=mode, title=title, context=context)

        # In ARR-LHRM doc: training sample is (q, m, c, a)
        # Here we encode it in the output text via tags so the model learns the 3 behaviors.
        if mode == "direct":
            answer_part = f"<no_think> Final answer: {answer_text}"
        elif mode == "retrieve":
            answer_part = (
                f" Related document title: {title}\n"
                f" Related document content: {context}\n\n"
                f"<no_think> Final answer: {answer_text}"
            )
        else:
            # Optional: if your QA has a reasoning trace field, you can inject it here.
            answer_part = f"<think>\n</think>\nFinal answer: {answer_text}"

        out.append([[prompt, answer_part], [False, True]])
        router_labels.append({"query": query, "mode": mode})

    json.dump(out, open(args.output_path, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", len(out), "instances to", args.output_path)
    if args.router_labels_output:
        with open(args.router_labels_output, "w", encoding="utf-8") as f:
            for obj in router_labels:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print("saved", len(router_labels), "router labels to", args.router_labels_output)


if __name__ == "__main__":
    main()

