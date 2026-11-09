"""Fine-tuning data curation: investigation traces -> SFT examples.

The idea (from the JD): fine-tune the diagnostic model on *real
investigation traces* — tool-call trajectories plus the final diagnosis
— instead of synthetic benchmarks. This module turns recorded
trajectories into JSONL training examples and ships the LoRA / QLoRA /
full-SFT configs for the actual GPU run.

What runs here (CPU): curation only. The training script
(``train_sft.py``) is a thin wrapper around HF Trainer + PEFT that
refuses to run without torch/peft/transformers and a GPU — it never
pretends to train on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingExample:
    instruction: str
    input: str  # tool-call trajectory + observations
    output: str  # diagnosis + recommended action

    def to_dict(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


INSTRUCTION = (
    "You are a network-intelligence diagnostic agent. Given an incident "
    "and the observations from your investigation tools, write the root "
    "cause, the evidence, and the recommended action."
)


def trace_to_example(trace: dict) -> TrainingExample:
    """One recorded investigation trajectory -> one SFT example."""
    incident = trace.get("incident", {})
    steps = trace.get("steps", [])
    lines = []
    for step in steps:
        lines.append(f"TOOL {step.get('tool')}: args={json.dumps(step.get('args', {}))}")
        obs = step.get("observation", "")
        lines.append(f"OBS: {obs}")
    input_text = (
        f"Incident {incident.get('id', '?')}: {incident.get('title', '')} "
        f"on {incident.get('device', '')}. Signals: {json.dumps(incident.get('signals', {}))}\n"
        + "\n".join(lines)
    )
    diagnosis = trace.get("diagnosis", {})
    output_text = (
        f"Root cause: {diagnosis.get('root_cause', 'unknown')}\n"
        f"Evidence: {'; '.join(diagnosis.get('evidence', []))}\n"
        f"Recommended action: {diagnosis.get('recommended_action', '')}"
    )
    return TrainingExample(instruction=INSTRUCTION, input=input_text, output=output_text)


def curate_dataset(traces_path: str | Path, out_path: str | Path) -> dict:
    """Read a traces JSONL file, write a training JSONL file.

    Returns stats: how many traces in, how many examples out, and how
    many were dropped (missing diagnosis or empty trajectory).
    """
    traces_path, out_path = Path(traces_path), Path(out_path)
    n_in = n_out = n_dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            trace = json.loads(line)
            if not trace.get("diagnosis") or not trace.get("steps"):
                n_dropped += 1
                continue
            fout.write(json.dumps(trace_to_example(trace).to_dict()) + "\n")
            n_out += 1
    return {"traces_in": n_in, "examples_out": n_out, "dropped": n_dropped, "out_path": str(out_path)}


def lora_config() -> dict:
    """PEFT LoRA config matching ``configs/lora_sft.yaml``."""
    return {
        "peft": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "task_type": "CAUSAL_LM"},
        "training": {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4,
                     "learning_rate": 2e-4, "num_train_epochs": 3, "bf16": True,
                     "logging_steps": 10, "save_steps": 100},
    }


def qlora_config() -> dict:
    """QLoRA config matching ``configs/qlora.yaml`` (4-bit base model)."""
    cfg = lora_config()
    cfg["quantization"] = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                           "bnb_4bit_compute_dtype": "bfloat16"}
    cfg["training"]["per_device_train_batch_size"] = 2
    return cfg
