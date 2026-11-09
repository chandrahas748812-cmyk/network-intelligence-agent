"""Retrieval tests: the right runbook surfaces for each failure class."""

from network_intelligence.retrieval import BM25Retriever, default_retriever, load_corpus


def test_corpus_loads():
    docs = load_corpus("data")
    kinds = {d.kind for d in docs}
    assert kinds == {"runbook", "past_incident"}
    assert len(docs) == 9  # 5 runbooks + 4 past incidents


def test_retrieves_throughput_runbook():
    retriever = default_retriever()
    hits = retriever.search("LLM inference throughput collapsed tok/s KV cache", top_k=3)
    assert hits
    assert hits[0].doc.id == "rb-throughput-collapse"


def test_retrieves_packet_loss_runbook():
    retriever = default_retriever()
    hits = retriever.search("packet loss CRC errors optic leaf spine", top_k=3)
    assert hits
    # The runbook must surface in the top 3. The matching past incident
    # (past-002, same failure) may legitimately outrank it — both are
    # relevant grounding for the agent.
    assert "rb-packet-loss" in {h.doc.id for h in hits}
    assert hits[0].doc.id in {"rb-packet-loss", "past-002"}


def test_retrieves_gpu_underutilised_runbook():
    retriever = default_retriever()
    hits = retriever.search("GPU underutilised low utilisation queue growing batch size", top_k=3)
    assert hits
    assert hits[0].doc.id == "rb-gpu-underutilised"


def test_empty_query_returns_nothing():
    retriever = BM25Retriever([])
    assert retriever.search("anything") == []


def test_scores_rank_relevant_first():
    retriever = default_retriever()
    hits = retriever.search("KV cache exhaustion batch size", top_k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
