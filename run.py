from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------- ARR-LHRM (3-mode) --------
    p_arr = sub.add_parser("arr", help="ARR-LHRM 3-mode inference (direct/retrieve/reason)")
    p_arr.add_argument("--query", type=str, required=True)
    p_arr.add_argument("--gen_model_path", type=str, required=True)
    p_arr.add_argument("--retrieval_model_path", type=str, required=True)
    p_arr.add_argument("--trie", type=str, required=True)
    p_arr.add_argument("--corpus", type=str, required=True)
    p_arr.add_argument("--batch_size", type=int, default=12)
    p_arr.add_argument("--candidate_num", type=int, default=0)
    p_arr.add_argument("--beam_num_for_title", type=int, default=5)
    p_arr.add_argument("--beam_num_for_chunk", type=int, default=10)
    p_arr.add_argument("--router_path", type=str, default="")
    p_arr.add_argument("--high_threshold", type=float, default=0.8)
    p_arr.add_argument("--low_threshold", type=float, default=0.3)

    # -------- Hybrid (2-stage proactive search) --------
    p_hy = sub.add_parser("hybrid", help="2-stage proactive retrieval (internal first, then optional external)")
    p_hy.add_argument("--query", type=str, required=True)
    p_hy.add_argument("--model_path", type=str, required=True)
    p_hy.add_argument("--trie", type=str, required=True)
    p_hy.add_argument("--corpus", type=str, required=True)
    p_hy.add_argument("--batch_size", type=int, default=12)
    p_hy.add_argument("--candidate_num", type=int, default=0)
    p_hy.add_argument("--beam_num_for_title", type=int, default=5)
    p_hy.add_argument("--beam_num_for_chunk", type=int, default=10)
    p_hy.add_argument("--high_threshold", type=float, default=0.8)
    p_hy.add_argument("--low_threshold", type=float, default=0.3)
    p_hy.add_argument("--min_margin_for_internal", type=float, default=0.0)
    p_hy.add_argument("--uncertain_triggers_external", action="store_true")
    p_hy.add_argument("--enable_local_kb", action="store_true")
    p_hy.add_argument("--local_kb_topk", type=int, default=3)
    p_hy.add_argument("--local_kb_accept_threshold", type=float, default=0.25)
    p_hy.add_argument("--external_provider", type=str, default="dummy")
    p_hy.add_argument("--external_topk", type=int, default=5)
    p_hy.add_argument("--enable_answer_generation", action="store_true")
    p_hy.add_argument("--answer_model_path", type=str, default="")
    p_hy.add_argument("--answer_max_new_tokens", type=int, default=128)

    args = p.parse_args()

    if args.cmd == "arr":
        from arr_lhrm.core import ARRConfig, SelfRetrievalConfig, run_arr_lhrm
        from arr_lhrm.router import RouterModel

        self_cfg = SelfRetrievalConfig(
            model_path=args.retrieval_model_path,
            trie_path=args.trie,
            corpus_path=args.corpus,
            batch_size=args.batch_size,
            candidate_num=args.candidate_num,
            beam_num_for_title=args.beam_num_for_title,
            beam_num_for_chunk=args.beam_num_for_chunk,
        )
        cfg = ARRConfig(
            gen_model_path=args.gen_model_path,
            self_retrieval=self_cfg,
            fallback_high_threshold=args.high_threshold,
            fallback_low_threshold=args.low_threshold,
        )
        router = RouterModel.load(args.router_path) if args.router_path else None
        out = run_arr_lhrm(query=args.query, cfg=cfg, router=router)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "hybrid":
        from arr_lhrm.core import HybridConfig, SelfRetrievalConfig, run_hybrid

        self_cfg = SelfRetrievalConfig(
            model_path=args.model_path,
            trie_path=args.trie,
            corpus_path=args.corpus,
            batch_size=args.batch_size,
            candidate_num=args.candidate_num,
            beam_num_for_title=args.beam_num_for_title,
            beam_num_for_chunk=args.beam_num_for_chunk,
        )
        cfg = HybridConfig(
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            min_margin_for_internal=args.min_margin_for_internal,
            uncertain_triggers_external=bool(args.uncertain_triggers_external),
            enable_local_kb=bool(args.enable_local_kb),
            local_kb_topk=args.local_kb_topk,
            local_kb_accept_threshold=args.local_kb_accept_threshold,
            external_provider=args.external_provider,
            external_topk=args.external_topk,
            enable_answer_generation=bool(args.enable_answer_generation),
            answer_model_path=args.answer_model_path,
            answer_max_new_tokens=args.answer_max_new_tokens,
        )
        out = run_hybrid(query=args.query, self_retrieval_cfg=self_cfg, hybrid_cfg=cfg)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

