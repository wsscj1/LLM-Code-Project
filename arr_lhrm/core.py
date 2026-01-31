from __future__ import annotations

"""
ARR-LHRM core (merged module).

This file consolidates previously split modules:
- modes / prompts / generate
- corpus utils
- confidence + decision policy
- local KB (TF-IDF / overlap)
- external web search clients
- Self-Retrieval wrapper (constrained generation + self-assessment)
- 3-mode ARR-LHRM inference
- 2-stage proactive hybrid pipeline
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import math
import os
import re
import urllib.parse
import urllib.request


# -------------------------
# Modes + prompts
# -------------------------


class Mode(str, Enum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"
    REASON = "reason"


@dataclass(frozen=True)
class Evidence:
    title: str
    content: str
    ref: str
    score: float = 0.0


def build_prompt(query: str, mode: Mode, evidences: Optional[List[Evidence]] = None) -> str:
    evidences = evidences or []
    header = f"Question: {query}\n\n"
    ctx = ""
    for ev in evidences:
        ctx += f" Related document title: {ev.title}\n Related document content: {ev.content}\n\n"

    if mode == Mode.DIRECT:
        return header + "<no_think> Final answer:"
    if mode == Mode.RETRIEVE:
        return header + ctx + "<no_think> Final answer:"
    if mode == Mode.REASON:
        # Encourage explicit reasoning traces (Hybrid-Reasoning style).
        return header + ctx + "<think>\n"
    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Generation helper
# -------------------------


@dataclass(frozen=True)
class GenerateConfig:
    model_path: str
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


def generate_text(prompt: str, cfg: GenerateConfig) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tok([prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample)
    if cfg.do_sample:
        gen_kwargs.update(dict(temperature=cfg.temperature, top_p=cfg.top_p))

    out = model.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


# -------------------------
# Corpus utilities
# -------------------------


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    return json.load(open(corpus_path, encoding="utf-8"))


def build_chunk_map(corpus: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    m: Dict[str, Tuple[str, str]] = {}
    for doc in corpus:
        wiki_id = str(doc["wikipedia_id"])
        title = str(doc["title"])
        for i, chunk in enumerate(doc["chunks"]):
            m[f"{wiki_id}.{i}"] = (title, str(chunk))
    return m


# -------------------------
# Confidence
# -------------------------


@dataclass(frozen=True)
class RetrievalConfidence:
    top1_prob: float
    top2_prob: float
    margin: float
    entropy: float
    ranked_chunks: List[Tuple[str, float]]


def _softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps)
    if s == 0:
        return [1.0 / len(xs) for _ in xs]
    return [e / s for e in exps]


def _entropy(probs: Sequence[float]) -> float:
    ent = 0.0
    for p in probs:
        if p <= 0:
            continue
        ent -= p * math.log(p + 1e-12)
    return ent


def rank_chunks_from_generation_instance(
    instance: Dict[str, Any],
    *,
    p_title_score: float = 0.4,
    p_selfassesment_score: float = 0.4,
) -> List[Tuple[str, float]]:
    title_scores: List[float] = list(instance.get("title_scores", []))
    title_generation: List[str] = list(instance.get("title_generation", []))
    documentids: Dict[str, List[Any]] = dict(instance.get("documentids", {}))
    model_scores: Dict[str, List[float]] = dict(instance.get("model_scores", {}))

    valid_title_scores: List[float] = []
    valid_titles: List[str] = []
    for i, t in enumerate(title_generation):
        try:
            s = float(title_scores[i])
        except Exception:
            continue
        if s <= -1e9:
            continue
        valid_titles.append(t)
        valid_title_scores.append(s)

    if not valid_titles:
        return []

    title_probs = _softmax([math.exp(s) for s in valid_title_scores], temperature=max(p_title_score, 1e-6))

    alldocs: Dict[str, float] = {}
    for title_idx, title in enumerate(valid_titles):
        doc_prob = title_probs[title_idx]
        chunk_ids = documentids.get(title, [])
        sa_scores = model_scores.get(title, [])

        if not sa_scores:
            per = 1.0 / max(1, len(chunk_ids))
            for cid in chunk_ids:
                if isinstance(cid, str):
                    alldocs[cid] = alldocs.get(cid, 0.0) + doc_prob * per
            continue

        transformed = [(1.0 - math.exp(-float(x))) for x in sa_scores]
        chunk_probs = _softmax(transformed, temperature=max(p_selfassesment_score, 1e-6))

        for j, cid in enumerate(chunk_ids[: len(chunk_probs)]):
            if not isinstance(cid, str):
                continue
            alldocs[cid] = alldocs.get(cid, 0.0) + doc_prob * chunk_probs[j]

    return sorted(alldocs.items(), key=lambda kv: kv[1], reverse=True)


def confidence_from_generation_instance(
    instance: Dict[str, Any],
    *,
    p_title_score: float = 0.4,
    p_selfassesment_score: float = 0.4,
    topk: int = 50,
) -> RetrievalConfidence:
    ranked = rank_chunks_from_generation_instance(
        instance,
        p_title_score=p_title_score,
        p_selfassesment_score=p_selfassesment_score,
    )[:topk]

    probs = [p for _, p in ranked]
    s = sum(probs)
    if s > 0:
        ranked = [(cid, p / s) for cid, p in ranked]
        probs = [p / s for p in probs]

    top1 = probs[0] if len(probs) >= 1 else 0.0
    top2 = probs[1] if len(probs) >= 2 else 0.0
    margin = top1 - top2
    ent = _entropy(probs)
    return RetrievalConfidence(
        top1_prob=float(top1),
        top2_prob=float(top2),
        margin=float(margin),
        entropy=float(ent),
        ranked_chunks=ranked,
    )


# -------------------------
# Decision
# -------------------------


class Action(str, Enum):
    USE_INTERNAL = "use_internal"
    TRIGGER_EXTERNAL = "trigger_external"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


@dataclass(frozen=True)
class ThresholdDecisionPolicy:
    high_threshold: float = 0.8
    low_threshold: float = 0.3
    uncertain_triggers_external: bool = True
    min_margin_for_internal: float = 0.0

    def decide(self, conf: RetrievalConfidence) -> Decision:
        if conf.top1_prob >= self.high_threshold and conf.margin >= self.min_margin_for_internal:
            return Decision(
                action=Action.USE_INTERNAL,
                reason=f"top1_prob={conf.top1_prob:.3f} >= high_threshold={self.high_threshold:.3f}",
            )
        if conf.top1_prob <= self.low_threshold:
            return Decision(
                action=Action.TRIGGER_EXTERNAL,
                reason=f"top1_prob={conf.top1_prob:.3f} <= low_threshold={self.low_threshold:.3f}",
            )
        if self.uncertain_triggers_external:
            return Decision(
                action=Action.TRIGGER_EXTERNAL,
                reason=f"uncertainty zone (top1_prob={conf.top1_prob:.3f}), trigger external by policy",
            )
        return Decision(
            action=Action.USE_INTERNAL,
            reason=f"uncertainty zone (top1_prob={conf.top1_prob:.3f}), keep internal by policy",
        )


# -------------------------
# Local KB index (TF-IDF / overlap)
# -------------------------


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    title: str
    chunk: str
    score: float


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _simple_tokens(s: str) -> List[str]:
    s = _normalize_text(s)
    return re.findall(r"[a-z0-9]+", s)


class LocalHybridIndex:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self._items: List[Tuple[str, str, str]] = []
        self._tfidf = False
        self._vectorizer = None
        self._matrix = None
        self._load_corpus()
        self._try_build_tfidf()

    def _load_corpus(self) -> None:
        corpus = load_corpus(self.corpus_path)
        items: List[Tuple[str, str, str]] = []
        for doc in corpus:
            wiki_id = str(doc["wikipedia_id"])
            title = str(doc["title"])
            for i, chunk in enumerate(doc["chunks"]):
                items.append((f"{wiki_id}.{i}", title, str(chunk)))
        self._items = items

    def _try_build_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception:
            return
        docs = [f"{title}\n{chunk}" for _, title, chunk in self._items]
        self._vectorizer = TfidfVectorizer(lowercase=True, max_features=200_000, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(docs)
        self._tfidf = True

    def search(self, query: str, topk: int = 5) -> List[VectorHit]:
        if not self._items or topk <= 0:
            return []
        if self._tfidf:
            qv = self._vectorizer.transform([query])  # type: ignore[union-attr]
            scores = (self._matrix @ qv.T).toarray().reshape(-1)  # type: ignore[union-attr]
            idxs = scores.argsort()[::-1][:topk]
            return [
                VectorHit(
                    chunk_id=self._items[int(i)][0],
                    title=self._items[int(i)][1],
                    chunk=self._items[int(i)][2],
                    score=float(scores[int(i)]),
                )
                for i in idxs
            ]

        qtokens = set(_simple_tokens(query))
        if not qtokens:
            return []
        scored: List[Tuple[float, int]] = []
        for idx, (_, title, chunk) in enumerate(self._items):
            dtokens = set(_simple_tokens(title + " " + chunk))
            inter = len(qtokens & dtokens)
            union = len(qtokens | dtokens)
            score = inter / union if union else 0.0
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits: List[VectorHit] = []
        for score, idx in scored[:topk]:
            cid, title, chunk = self._items[idx]
            hits.append(VectorHit(chunk_id=cid, title=title, chunk=chunk, score=float(score)))
        return hits


# -------------------------
# External search
# -------------------------


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class ExternalSearchClient:
    def search(self, query: str, topk: int = 5) -> List[WebResult]:
        raise NotImplementedError


class DummySearchClient(ExternalSearchClient):
    def search(self, query: str, topk: int = 5) -> List[WebResult]:
        return []


class BingWebSearchClient(ExternalSearchClient):
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BING_SEARCH_KEY", "")
        self.endpoint = endpoint or os.environ.get("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
        if not self.api_key:
            raise ValueError("Missing Bing API key. Set BING_SEARCH_KEY.")

    def search(self, query: str, topk: int = 5) -> List[WebResult]:
        params = {"q": query, "count": str(topk), "textDecorations": "false", "textFormat": "Raw"}
        url = self.endpoint + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        web_pages = (data or {}).get("webPages", {}).get("value", []) or []
        results: List[WebResult] = []
        for item in web_pages:
            results.append(
                WebResult(
                    title=str(item.get("name", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                    score=float(item.get("rank", 0.0) or 0.0),
                )
            )
        return results


def build_external_client(provider: str) -> ExternalSearchClient:
    provider = (provider or "").strip().lower()
    if provider in {"", "none", "off", "offline", "dummy"}:
        return DummySearchClient()
    if provider in {"bing", "bing_web"}:
        return BingWebSearchClient()
    raise ValueError(f"Unknown external search provider: {provider}")


# -------------------------
# Self-Retrieval wrapper
# -------------------------


@dataclass(frozen=True)
class SelfRetrievalConfig:
    model_path: str
    trie_path: str
    corpus_path: str
    batch_size: int = 12
    candidate_num: int = 0
    beam_num_for_title: int = 5
    beam_num_for_chunk: int = 10
    num_for_chunk_generation_stage1: int = 1


def _build_chunkid_2_title_chunk(corpus: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    m: Dict[str, Tuple[str, str]] = {}
    for doc in corpus:
        wiki_id = str(doc["wikipedia_id"])
        title = str(doc["title"])
        for c, chunk in enumerate(doc["chunks"]):
            m[f"{wiki_id}.{c}"] = (title, str(chunk))
    return m


def _decide_sequences(tokenizer) -> Tuple[str, str, str]:
    in_context_token = tokenizer.encode(": title", add_special_tokens=False)
    with_space_token = tokenizer.encode(" title", add_special_tokens=False)
    without_space_token = tokenizer.encode("title", add_special_tokens=False)
    if in_context_token[1] == with_space_token[0]:
        return (" Related document title:", " cannot extract answer", " Related document content:")
    if in_context_token[1] == without_space_token[0]:
        return ("Related document title:", "cannot extract answer", "Related document content:")
    raise ValueError("Can't decide if need add space for this tokenizer")


def _as_generation_instance(result: Dict[str, Any], beam_num_for_chunk: int) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["id"] = result.get("id", "")
    data["query"] = result.get("query", "")
    data["answers"] = result.get("answers", [])

    predicted_titles = list(result.get("predicted_titles", []))
    data["title_scores"] = [t[1] for t in predicted_titles]
    data["title_generation"] = [t[0] for t in predicted_titles]

    predicted_1_chunks = list(result.get("predicted_1_chunks", []))
    predicted_extra_chunks = list(result.get("predicted_extra_chunks", []))
    predicted_self_assessment_score = list(result.get("predicted_self_assessment_score", []))

    chunk_scores = [c[1] for c in predicted_1_chunks]
    chunk_ids = [c[0] for c in predicted_extra_chunks]

    data["chunk_scores"] = []
    data["documentids"] = {}
    data["model_scores"] = {}
    for t, title in enumerate(data["title_generation"]):
        start = t * beam_num_for_chunk
        end = (t + 1) * beam_num_for_chunk
        data["chunk_scores"].append(chunk_scores[start:end])
        data["documentids"][title] = chunk_ids[start:end]
        data["model_scores"][title] = predicted_self_assessment_score[start:end]
    return data


def run_self_retrieval_single_query(cfg: SelfRetrievalConfig, query: str) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from evaluation.predict import Retriever  # type: ignore

    trie = json.load(open(cfg.trie_path, encoding="utf-8"))
    corpus = load_corpus(cfg.corpus_path)
    chunkid_2_title_chunk = _build_chunkid_2_title_chunk(corpus)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    start_sequence, self_assessment_sequence, prompt_for_chunk = _decide_sequences(tokenizer)
    start_token_ids = tokenizer(start_sequence, add_special_tokens=False)["input_ids"]
    _ = {str(tokenizer.encode(start_sequence)[-1]): trie}

    retriever = Retriever(
        constrain_dict=trie,
        chunkid_2_title_chunk={k: [v[0], v[1]] for k, v in chunkid_2_title_chunk.items()},
        start_token_ids=start_token_ids,
        self_assesment_sequence=self_assessment_sequence,
        prompt_for_chunk=prompt_for_chunk,
        batch_size=cfg.batch_size,
        candidate_num=cfg.candidate_num,
        beam_num_for_title=cfg.beam_num_for_title,
        beam_num_for_chunk=cfg.beam_num_for_chunk,
        num_for_chunk_generation_stage1=cfg.num_for_chunk_generation_stage1,
    )

    results = retriever.run([{"id": "q0", "query": query, "answers": []}], cfg.model_path)
    if not results:
        return {
            "id": "q0",
            "query": query,
            "answers": [],
            "title_scores": [],
            "title_generation": [],
            "chunk_scores": [],
            "documentids": {},
            "model_scores": {},
        }
    return _as_generation_instance(results[0], cfg.beam_num_for_chunk)


# -------------------------
# 2-stage hybrid pipeline (proactive external search)
# -------------------------


@dataclass(frozen=True)
class HybridConfig:
    high_threshold: float = 0.8
    low_threshold: float = 0.3
    uncertain_triggers_external: bool = True
    min_margin_for_internal: float = 0.0

    internal_topk: int = 3

    enable_local_kb: bool = True
    local_kb_topk: int = 3
    local_kb_accept_threshold: float = 0.25

    external_provider: str = "dummy"
    external_topk: int = 5

    enable_answer_generation: bool = False
    answer_model_path: str = ""
    answer_max_new_tokens: int = 128


def run_hybrid(*, query: str, self_retrieval_cfg: SelfRetrievalConfig, hybrid_cfg: HybridConfig) -> Dict[str, Any]:
    corpus = load_corpus(self_retrieval_cfg.corpus_path)
    chunk_map = build_chunk_map(corpus)

    generation_instance = run_self_retrieval_single_query(self_retrieval_cfg, query)
    conf = confidence_from_generation_instance(generation_instance)

    policy = ThresholdDecisionPolicy(
        high_threshold=hybrid_cfg.high_threshold,
        low_threshold=hybrid_cfg.low_threshold,
        uncertain_triggers_external=hybrid_cfg.uncertain_triggers_external,
        min_margin_for_internal=hybrid_cfg.min_margin_for_internal,
    )
    decision = policy.decide(conf)

    evidences: List[Dict[str, Any]] = []
    for cid, prob in conf.ranked_chunks[: max(0, hybrid_cfg.internal_topk)]:
        if cid in chunk_map:
            title, chunk = chunk_map[cid]
            evidences.append(
                {"source": "self_retrieval", "title": title, "ref": cid, "score": float(prob), "content": chunk}
            )

    used_external = False
    used_local_kb = False

    if decision.action.value == "trigger_external":
        if hybrid_cfg.enable_local_kb:
            idx = LocalHybridIndex(self_retrieval_cfg.corpus_path)
            local_hits = idx.search(query, topk=hybrid_cfg.local_kb_topk)
            if local_hits and local_hits[0].score >= hybrid_cfg.local_kb_accept_threshold:
                used_local_kb = True
                for h in local_hits:
                    evidences.append(
                        {
                            "source": "local_kb",
                            "title": h.title,
                            "ref": h.chunk_id,
                            "score": float(h.score),
                            "content": h.chunk,
                        }
                    )
            else:
                try:
                    client = build_external_client(hybrid_cfg.external_provider)
                    web_hits = client.search(query, topk=hybrid_cfg.external_topk)
                except Exception:
                    web_hits = []
                used_external = True if web_hits else False
                for w in web_hits:
                    evidences.append(
                        {"source": "web", "title": w.title, "ref": w.url, "score": float(w.score), "content": w.snippet}
                    )

    answer: Optional[str] = None
    if hybrid_cfg.enable_answer_generation and hybrid_cfg.answer_model_path:
        prompt = f"Question: {query}\n\n"
        for e in evidences:
            prompt += f" Related document title: {e['title']}\n Related document content: {e['content']}\n\n"
        prompt += " Final answer:"
        answer = generate_text(
            prompt,
            GenerateConfig(model_path=hybrid_cfg.answer_model_path, max_new_tokens=hybrid_cfg.answer_max_new_tokens, do_sample=False),
        )

    return {
        "query": query,
        "decision": {"action": decision.action.value, "reason": decision.reason},
        "arr_lhrm_confidence": {"top1_prob": conf.top1_prob, "top2_prob": conf.top2_prob, "margin": conf.margin, "entropy": conf.entropy},
        "used_local_kb": used_local_kb,
        "used_external": used_external,
        "evidence": evidences,
        "answer": answer,
    }


# -------------------------
# 3-mode ARR-LHRM inference
# -------------------------


@dataclass(frozen=True)
class ARRConfig:
    gen_model_path: str
    self_retrieval: SelfRetrievalConfig
    fallback_high_threshold: float = 0.8
    fallback_low_threshold: float = 0.3
    always_run_internal_retrieval_for_logging: bool = False
    direct_max_new_tokens: int = 128
    reason_max_new_tokens: int = 256
    retrieve_max_new_tokens: int = 128


def _mode_from_router_index(idx: int) -> Mode:
    return {0: Mode.DIRECT, 1: Mode.RETRIEVE, 2: Mode.REASON}[idx]


def run_arr_lhrm(*, query: str, cfg: ARRConfig, router: Optional[Any] = None) -> Dict[str, Any]:
    # Decide mode
    if router is not None:
        mode = _mode_from_router_index(int(router.predict(query)))
        route_reason = "router_policy"
    else:
        generation_instance = run_self_retrieval_single_query(cfg.self_retrieval, query)
        conf = confidence_from_generation_instance(generation_instance)
        pol = ThresholdDecisionPolicy(
            high_threshold=cfg.fallback_high_threshold,
            low_threshold=cfg.fallback_low_threshold,
            uncertain_triggers_external=True,
        )
        d = pol.decide(conf)
        mode = Mode.RETRIEVE if d.action.value == "use_internal" else Mode.REASON
        route_reason = f"threshold_fallback({d.reason})"

    generation_instance: Optional[Dict[str, Any]] = None
    conf: Optional[RetrievalConfidence] = None
    evidences: List[Evidence] = []

    if cfg.always_run_internal_retrieval_for_logging or mode == Mode.RETRIEVE:
        generation_instance = run_self_retrieval_single_query(cfg.self_retrieval, query)
        conf = confidence_from_generation_instance(generation_instance)
        chunk_map = build_chunk_map(load_corpus(cfg.self_retrieval.corpus_path))
        for cid, p in conf.ranked_chunks[:3]:
            if cid in chunk_map:
                t, ck = chunk_map[cid]
                evidences.append(Evidence(title=t, content=ck, ref=cid, score=float(p)))

    if mode == Mode.DIRECT:
        prompt = build_prompt(query, Mode.DIRECT, [])
        ans = generate_text(prompt, GenerateConfig(model_path=cfg.gen_model_path, max_new_tokens=cfg.direct_max_new_tokens))
    elif mode == Mode.RETRIEVE:
        prompt = build_prompt(query, Mode.RETRIEVE, evidences)
        ans = generate_text(prompt, GenerateConfig(model_path=cfg.gen_model_path, max_new_tokens=cfg.retrieve_max_new_tokens))
    else:
        prompt = build_prompt(query, Mode.REASON, [])
        ans = generate_text(prompt, GenerateConfig(model_path=cfg.gen_model_path, max_new_tokens=cfg.reason_max_new_tokens))

    return {
        "query": query,
        "mode": mode.value,
        "route_reason": route_reason,
        "arr_lhrm_confidence": None
        if conf is None
        else {"top1_prob": conf.top1_prob, "top2_prob": conf.top2_prob, "margin": conf.margin, "entropy": conf.entropy},
        "answer": ans,
        "self_retrieval_generation": generation_instance,
    }

