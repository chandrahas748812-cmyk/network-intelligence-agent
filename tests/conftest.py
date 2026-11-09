"""Shared fixtures: seeded synthetic telemetry + a demo agent."""

import pytest

from network_intelligence.agent import build_agent
from network_intelligence.config import AgentConfig, ServingConfig
from network_intelligence.retrieval import default_retriever
from network_intelligence.serving import get_backend
from network_intelligence.stores import InMemoryIncidentStore, InMemoryMetricsStore
from network_intelligence.telemetry import TelemetryGenerator


@pytest.fixture()
def bundle():
    return TelemetryGenerator(seed=42).generate(minutes=60)


@pytest.fixture()
def metrics_store(bundle):
    store = InMemoryMetricsStore()
    store.load(bundle)
    return store


@pytest.fixture()
def incident_store(bundle):
    store = InMemoryIncidentStore()
    store.load(bundle)
    return store


@pytest.fixture()
def agent(metrics_store, incident_store):
    return build_agent(
        get_backend(ServingConfig()),
        metrics_store,
        incident_store,
        retriever=default_retriever(),
        cfg=AgentConfig(),
    )
