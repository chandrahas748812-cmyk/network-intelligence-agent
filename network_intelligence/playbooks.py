"""Structured investigation playbooks.

A playbook is the written procedure for one failure class: what to
check, in what order, and what the checks mean. The agent selects a
playbook from the incident's signals, then executes its steps through
the tool runner. Playbooks live in code (so tests can pin them) and
their human-readable twins live in ``data/runbooks.json`` (so the
retriever can ground the agent in them).

Playbooks do NOT invent diagnoses — they order the evidence gathering.
The diagnosis itself comes from ``decision_tree`` + the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlaybookStep:
    tool: str
    args: dict
    why: str


@dataclass
class Playbook:
    id: str
    name: str
    triggers: list[str]  # signal keys whose presence suggests this playbook
    steps: list[PlaybookStep]
    remediation: str


PLAYBOOKS: list[Playbook] = [
    Playbook(
        id="pb-throughput-collapse",
        name="LLM inference throughput collapse",
        triggers=["inference_tok_per_s", "kv_cache_usage_pct"],
        steps=[
            PlaybookStep("query_metrics", {"metrics": ["inference_tok_per_s", "kv_cache_usage_pct", "p99_latency_ms", "gpu_util_pct"], "minutes": 30}, "Confirm the throughput drop and cache pressure."),
            PlaybookStep("search_logs", {"level": "ERROR", "limit": 20}, "Look for KV-cache allocation failures or OOM guards."),
            PlaybookStep("check_alerts", {}, "See if the cache / throughput alerts already fired."),
            PlaybookStep("correlate_events", {"minutes": 30}, "Check alerts and errors line up in time."),
            PlaybookStep("decision_tree", {}, "Classify: KV exhaustion vs batching vs something else."),
        ],
        remediation="Free KV headroom (raise kv_cache_gb or lower max_num_seqs), then re-tune batch size; verify tok/s recovers before closing.",
    ),
    Playbook(
        id="pb-packet-loss",
        name="Packet loss on a leaf-spine link",
        triggers=["packet_loss_pct", "interface_errors"],
        steps=[
            PlaybookStep("query_metrics", {"metrics": ["packet_loss_pct", "interface_errors", "p99_latency_ms"], "minutes": 30}, "Quantify loss and error rate."),
            PlaybookStep("search_logs", {"level": "ERROR", "limit": 20}, "Look for CRC errors, flaps, or link-down events."),
            PlaybookStep("check_alerts", {}, "See if the loss alert already fired."),
            PlaybookStep("correlate_events", {"minutes": 30}, "Check errors and alerts share the same interface and window."),
            PlaybookStep("decision_tree", {}, "Classify: faulty optic vs congestion vs config."),
        ],
        remediation="Drain the link, replace the optic / reseat the cable, then watch CRC counters for 30m before returning it to service.",
    ),
    Playbook(
        id="pb-gpu-underutilised",
        name="GPU underutilised while queue grows",
        triggers=["gpu_util_pct"],
        steps=[
            PlaybookStep("query_metrics", {"metrics": ["gpu_util_pct", "inference_tok_per_s", "p99_latency_ms"], "minutes": 30}, "Confirm low util against queue depth."),
            PlaybookStep("search_logs", {"level": "WARN", "limit": 20}, "Look for scheduler / batching warnings."),
            PlaybookStep("threshold_check", {"metric": "gpu_util_pct", "threshold": 20.0, "op": "lt"}, "Formal threshold check for the report."),
            PlaybookStep("decision_tree", {}, "Classify: batching too small vs model too small vs scheduler cap."),
        ],
        remediation="Raise max_num_seqs / batch size until GPU util and tok/s both rise; if the queue still grows, add a replica.",
    ),
    Playbook(
        id="pb-generic",
        name="Generic investigation",
        triggers=[],
        steps=[
            PlaybookStep("query_metrics", {"metrics": [], "minutes": 30}, "Gather the incident's signal metrics."),
            PlaybookStep("search_logs", {"level": "ERROR", "limit": 20}, "Look for error lines around the incident window."),
            PlaybookStep("check_alerts", {}, "List active alerts on the device."),
            PlaybookStep("correlate_events", {"minutes": 30}, "Line alerts and errors up in time."),
            PlaybookStep("decision_tree", {}, "Classify from the signals."),
        ],
        remediation="Follow the decision-tree branch; escalate to the owning team with the evidence bundle if no branch fires.",
    ),
]


def select_playbook(incident: dict) -> Playbook:
    """Pick the playbook whose triggers best match the incident signals.

    Highest trigger overlap wins; ties go to the more specific playbook
    (fewest triggers), so e.g. low GPU util picks the underutilisation
    playbook over the general throughput one.
    """
    signals = set(incident.get("signals", {}))
    scored = [
        (len(signals & set(pb.triggers)), -len(pb.triggers), pb)
        for pb in PLAYBOOKS[:-1]
    ]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_score, _, best = scored[0]
    return best if best_score > 0 else PLAYBOOKS[-1]


def playbook_for_root_cause(root_cause: str) -> Playbook | None:
    mapping = {
        "kv_cache_exhaustion": "pb-throughput-collapse",
        "faulty_optic": "pb-packet-loss",
        "batching_too_small": "pb-gpu-underutilised",
    }
    pid = mapping.get(root_cause)
    for pb in PLAYBOOKS:
        if pb.id == pid:
            return pb
    return None
