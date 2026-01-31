from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: str, max_n: int = 0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if max_n and len(out) >= max_n:
                break
    return out


def cmd_hft(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    from arr_lhrm.router import RouterConfig, RouterModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mode2id = {"direct": 0, "retrieve": 1, "reason": 2}
    data = _load_jsonl(args.train_jsonl)
    queries: List[str] = []
    labels: List[int] = []
    for obj in data:
        q = str(obj.get("query", "")).strip()
        m = str(obj.get("mode", "")).strip().lower()
        if not q or m not in mode2id:
            continue
        queries.append(q)
        labels.append(mode2id[m])
    if not queries:
        raise ValueError("No training samples. Expect jsonl lines with {query, mode}.")

    router = RouterModel(RouterConfig(base_model_name_or_path=args.base_model, max_length=args.max_length))
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr)

    n = len(queries)
    idxs = list(range(n))
    for epoch in range(args.epochs):
        random.shuffle(idxs)
        total_loss = 0.0
        total = 0
        correct = 0
        for i in range(0, n, args.batch_size):
            bidx = idxs[i : i + args.batch_size]
            bq = [queries[j] for j in bidx]
            bl = torch.tensor([labels[j] for j in bidx], device=router.device, dtype=torch.long)
            lg = router.logits(bq)
            loss = F.cross_entropy(lg, bl)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(bidx)
            total += len(bidx)
            pred = torch.argmax(lg, dim=-1)
            correct += int((pred == bl).sum().item())

        print(f"[router-hft] epoch={epoch+1} loss={total_loss/total:.4f} acc={correct/total:.4f} n={total}")

    os.makedirs(args.output_dir, exist_ok=True)
    router.save(args.output_dir)
    print("saved router to", args.output_dir)


def cmd_hgpo(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    from arr_lhrm.core import (
        Evidence,
        GenerateConfig,
        Mode,
        SelfRetrievalConfig,
        build_chunk_map,
        build_prompt,
        confidence_from_generation_instance,
        generate_text,
        load_corpus,
        run_self_retrieval_single_query,
    )
    from arr_lhrm.reward import RewardConfig, compute_reward
    from arr_lhrm.router import RouterModel

    random.seed(args.seed)

    router = RouterModel.load(args.router_path)
    train = _load_jsonl(args.train_jsonl, max_n=args.max_train)
    if not train:
        raise ValueError("Empty train_jsonl")

    gen_cfg = GenerateConfig(
        model_path=args.gen_model,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    reward_cfg = RewardConfig()
    self_cfg = SelfRetrievalConfig(model_path=args.retrieval_model, trie_path=args.trie, corpus_path=args.corpus)

    corpus = load_corpus(args.corpus)
    chunk_map = build_chunk_map(corpus)

    opt = torch.optim.AdamW(router.parameters(), lr=args.lr)

    def retrieve_evidence_and_relevance(query: str) -> Tuple[List[Evidence], float]:
        gen_inst = run_self_retrieval_single_query(self_cfg, query)
        conf = confidence_from_generation_instance(gen_inst)
        evs: List[Evidence] = []
        for cid, p in conf.ranked_chunks[:3]:
            if cid in chunk_map:
                t, ck = chunk_map[cid]
                evs.append(Evidence(title=t, content=ck, ref=cid, score=float(p)))
        return evs, float(conf.top1_prob)

    def sample_candidates_for_query(q: str, gold: Optional[str]) -> Dict[str, Any]:
        res: Dict[str, Any] = {"query": q, "gold": gold, "modes": {}}
        retrieve_evs, rel = retrieve_evidence_and_relevance(q)
        for mode in [Mode.DIRECT, Mode.RETRIEVE, Mode.REASON]:
            rewards: List[float] = []
            texts: List[str] = []
            for _ in range(args.n_per_mode):
                prompt = build_prompt(q, mode, retrieve_evs if mode == Mode.RETRIEVE else [])
                txt = generate_text(prompt, gen_cfg)
                texts.append(txt)
                cost = len(txt.split())
                relevance = rel if mode == Mode.RETRIEVE else 0.0
                rewards.append(float(compute_reward(pred=txt, gold=gold, relevance=relevance, cost_tokens=cost, cfg=reward_cfg)))
            res["modes"][mode.value] = {"texts": texts, "rewards": rewards}
        return res

    def _avg(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    def best_mode(sample: Dict[str, Any]) -> str:
        avgs = {m: _avg(obj["rewards"]) for m, obj in sample["modes"].items()}
        ranked = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)
        best_m, best_v = ranked[0]
        if len(ranked) >= 2 and abs(best_v - ranked[1][1]) < float(args.delta):
            def cost_of(m: str) -> float:
                return _avg([len(t.split()) for t in sample["modes"][m]["texts"]])

            return min([best_m, ranked[1][0]], key=cost_of)
        return best_m

    mode2id = {"direct": 0, "retrieve": 1, "reason": 2}

    for epoch in range(args.epochs):
        random.shuffle(train)
        total_loss = 0.0
        total = 0
        correct = 0
        for i in range(0, len(train), args.batch_size):
            batch = train[i : i + args.batch_size]
            samples = [sample_candidates_for_query(obj["query"], obj.get("gold")) for obj in batch]
            targets = [best_mode(s) for s in samples]
            y = torch.tensor([mode2id[t] for t in targets], device=router.device, dtype=torch.long)

            logits = router.logits([obj["query"] for obj in batch])
            adv = torch.ones_like(y, dtype=torch.float, device=router.device) * float(args.alpha)
            loss = F.cross_entropy(logits, y, reduction="none")
            loss = (loss * adv).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(batch)
            total += len(batch)
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())

        print(f"[router-hgpo] epoch={epoch+1} loss={total_loss/total:.4f} hacc={correct/total:.4f} n={total}")

    os.makedirs(args.out_router_path, exist_ok=True)
    router.save(args.out_router_path)
    print("saved hgpo-trained router to", args.out_router_path)


def cmd_hacc(args: argparse.Namespace) -> None:
    from arr_lhrm.core import (
        Evidence,
        GenerateConfig,
        Mode,
        SelfRetrievalConfig,
        build_chunk_map,
        build_prompt,
        confidence_from_generation_instance,
        generate_text,
        load_corpus,
        run_self_retrieval_single_query,
    )
    from arr_lhrm.reward import RewardConfig, compute_reward
    from arr_lhrm.router import RouterModel

    random.seed(args.seed)
    router = RouterModel.load(args.router_path)
    data = _load_jsonl(args.eval_jsonl, max_n=args.max_eval)
    if not data:
        raise ValueError("Empty eval_jsonl")

    gen_cfg = GenerateConfig(model_path=args.gen_model, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95)
    reward_cfg = RewardConfig()
    self_cfg = SelfRetrievalConfig(model_path=args.retrieval_model, trie_path=args.trie, corpus_path=args.corpus)

    corpus = load_corpus(args.corpus)
    chunk_map = build_chunk_map(corpus)

    def retrieve_evidence_and_relevance(query: str) -> Tuple[List[Evidence], float]:
        gen_inst = run_self_retrieval_single_query(self_cfg, query)
        conf = confidence_from_generation_instance(gen_inst)
        evs: List[Evidence] = []
        for cid, p in conf.ranked_chunks[:3]:
            if cid in chunk_map:
                t, ck = chunk_map[cid]
                evs.append(Evidence(title=t, content=ck, ref=cid, score=float(p)))
        return evs, float(conf.top1_prob)

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    def ground_truth_mode(q: str, gold: Optional[str]) -> str:
        evs, rel = retrieve_evidence_and_relevance(q)
        avgs: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        for mode in [Mode.DIRECT, Mode.RETRIEVE, Mode.REASON]:
            rs: List[float] = []
            cs: List[int] = []
            for _ in range(args.n_per_mode):
                prompt = build_prompt(q, mode, evs if mode == Mode.RETRIEVE else [])
                txt = generate_text(prompt, gen_cfg)
                c = len(txt.split())
                cs.append(c)
                relevance = rel if mode == Mode.RETRIEVE else 0.0
                rs.append(float(compute_reward(pred=txt, gold=gold, relevance=relevance, cost_tokens=c, cfg=reward_cfg)))
            avgs[mode.value] = avg(rs)
            costs[mode.value] = avg(cs)

        ranked = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)
        best_m, best_v = ranked[0]
        if len(ranked) >= 2 and abs(best_v - ranked[1][1]) < float(args.delta):
            return min([best_m, ranked[1][0]], key=lambda m: costs.get(m, 1e9))
        return best_m

    total = 0
    match = 0
    for obj in data:
        q = str(obj.get("query", "")).strip()
        if not q:
            continue
        gt = ground_truth_mode(q, obj.get("gold"))
        pred_idx = router.predict(q)
        pred = {0: "direct", 1: "retrieve", 2: "reason"}[pred_idx]
        total += 1
        match += 1 if pred == gt else 0

    if total == 0:
        raise ValueError("No valid queries")
    print(f"HAcc={match/total:.4f} ({match}/{total})")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_hft = sub.add_parser("hft", help="Stage I: supervised router training")
    p_hft.add_argument("--base_model", type=str, required=True)
    p_hft.add_argument("--train_jsonl", type=str, required=True)
    p_hft.add_argument("--output_dir", type=str, required=True)
    p_hft.add_argument("--epochs", type=int, default=3)
    p_hft.add_argument("--batch_size", type=int, default=32)
    p_hft.add_argument("--lr", type=float, default=5e-4)
    p_hft.add_argument("--max_length", type=int, default=256)
    p_hft.add_argument("--seed", type=int, default=42)
    p_hft.set_defaults(func=cmd_hft)

    p_rl = sub.add_parser("hgpo", help="Stage II: HGPO-style router RL (lightweight)")
    p_rl.add_argument("--router_path", type=str, required=True)
    p_rl.add_argument("--gen_model", type=str, required=True)
    p_rl.add_argument("--train_jsonl", type=str, required=True, help="jsonl with fields: query, gold(optional)")
    p_rl.add_argument("--out_router_path", type=str, required=True)
    p_rl.add_argument("--max_train", type=int, default=0)
    p_rl.add_argument("--seed", type=int, default=42)
    p_rl.add_argument("--n_per_mode", type=int, default=2)
    p_rl.add_argument("--delta", type=float, default=0.0)
    p_rl.add_argument("--alpha", type=float, default=1.0)
    p_rl.add_argument("--lr", type=float, default=5e-4)
    p_rl.add_argument("--epochs", type=int, default=1)
    p_rl.add_argument("--batch_size", type=int, default=8)
    p_rl.add_argument("--max_new_tokens", type=int, default=128)
    p_rl.add_argument("--temperature", type=float, default=0.7)
    p_rl.add_argument("--top_p", type=float, default=0.95)
    p_rl.add_argument("--trie", type=str, required=True)
    p_rl.add_argument("--corpus", type=str, required=True)
    p_rl.add_argument("--retrieval_model", type=str, required=True)
    p_rl.set_defaults(func=cmd_hgpo)

    p_hacc = sub.add_parser("hacc", help="Evaluate Hybrid Accuracy (HAcc) for router")
    p_hacc.add_argument("--router_path", type=str, required=True)
    p_hacc.add_argument("--gen_model", type=str, required=True)
    p_hacc.add_argument("--eval_jsonl", type=str, required=True, help="jsonl with fields: query, gold(optional)")
    p_hacc.add_argument("--max_eval", type=int, default=0)
    p_hacc.add_argument("--seed", type=int, default=42)
    p_hacc.add_argument("--n_per_mode", type=int, default=2)
    p_hacc.add_argument("--delta", type=float, default=0.0)
    p_hacc.add_argument("--trie", type=str, required=True)
    p_hacc.add_argument("--corpus", type=str, required=True)
    p_hacc.add_argument("--retrieval_model", type=str, required=True)
    p_hacc.set_defaults(func=cmd_hacc)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

