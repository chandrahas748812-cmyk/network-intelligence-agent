"""Ray Serve sketch: distributed serving across GPU nodes.

This is a *sketch* for the GPU run — it is not executed by the demo and
Ray is not a demo dependency. It shows the shape the JD asks for:
a model deployment replicated across nodes, with autoscaling and a
router, plus the 1-GPU-vs-4-GPU decision made explicit in config
(``configs/ray_serve.yaml``).

On a GPU machine: ``pip install "ray[serve]"`` then
``serve run configs/ray_serve.yaml`` (or ``python -m
network_intelligence.ray_serve``).
"""

from __future__ import annotations

import os


def build_app():
    """Build the Ray Serve application. Importing ray is deferred so the
    rest of the package works without it."""
    try:
        from ray import serve  # type: ignore
        from ray.serve.handle import DeploymentHandle  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Ray is not installed. On a GPU machine: "
            'pip install "ray[serve]" — the CPU demo does not need it.'
        ) from exc

    model_id = os.environ.get("NIA_MODEL", "mock-diagnostic-llm-0.1")
    num_replicas = int(os.environ.get("NIA_REPLICAS", "1"))

    @serve.deployment(
        name="diagnostic-llm",
        num_replicas=num_replicas,
        ray_actor_options={"num_gpus": 1},
        autoscaling_config={"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 8},
    )
    class DiagnosticLLM:
        def __init__(self) -> None:
            # Real deployment: load the vLLM engine here (see serving.py).
            # The 1-vs-4 GPU question is answered by num_replicas +
            # tensor_parallel_size in ServingConfig, not by guessing.
            self.model_id = model_id

        async def __call__(self, request):  # noqa: ANN001, ANN202
            body = await request.json()
            return {
                "model": self.model_id,
                "note": "Ray Serve sketch — wire a real engine here.",
                "prompt_tokens": len(body.get("prompt", "")) // 4,
            }

    @serve.deployment(name="router")
    class Router:
        def __init__(self, handle) -> None:  # noqa: ANN001
            self.handle = handle

        async def __call__(self, request):  # noqa: ANN001, ANN202
            return await self.handle.remote(request)

    app = Router.bind(DiagnosticLLM.bind())
    return app


if __name__ == "__main__":
    from ray import serve  # type: ignore

    serve.run(build_app())
    print("Ray Serve app running. POST prompts to the router deployment.")
