"""Network Intelligence Agent — portfolio / learning project.

An investigation agent for LLM-inference infrastructure: it watches
telemetry, investigates incidents by calling tools step by step, retrieves
runbooks and past incidents, and writes structured diagnostic reports.

This is a from-scratch portfolio project built to demonstrate the skills
in the Tesla "Sr. ML Engineer, Network Intelligence" posting. It is not
employer work, contains no proprietary data, and all telemetry, incidents,
and runbooks are synthetic.
"""

__version__ = "0.1.0"

from network_intelligence.config import (
    AgentConfig,
    ServingConfig,
    WatcherConfig,
)

__all__ = ["__version__", "AgentConfig", "ServingConfig", "WatcherConfig"]
