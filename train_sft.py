"""SFT training entry point (GPU only).

Curates the dataset from ``data/traces.jsonl`` (CPU-safe) and then
trains with HF Trainer + PEFT LoRA/QLoRA. This script REFUSES to run
without torch, transformers, peft, and a CUDA GPU — it will not
pretend to train on CPU.

Usage (GPU machine):
    python train_sft.py --config configs/lora_sft.yaml
    python train_sft.py --config configs/qlora.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training for the diagnostic model (GPU only).")
    parser.add_argument("--config", default="configs/lora_sft.yaml")
    parser.add_argument("--traces", default="data/traces.jsonl")
    parser.add_argument("--out", default="data/sft_train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft-v0.1.0")
    args = parser.parse_args()

    try:
        import torch  # type: ignore
    except ImportError:
        print("ERROR: torch is not installed. Training needs the GPU stack:\n"
              "  pip install torch transformers peft datasets accelerate\n"
              "See README 'Running on a real GPU'.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU detected. Training is GPU-only by design; "
              "curation (finetune.curate_dataset) already ran on CPU.", file=sys.stderr)
        return 2
    try:
        import transformers  # noqa: F401  # type: ignore
        import peft  # noqa: F401  # type: ignore
    except ImportError as exc:
        print(f"ERROR: missing training dependency: {exc}. "
              "pip install transformers peft datasets accelerate", file=sys.stderr)
        return 2

    # Curation is CPU-safe and runs first.
    from network_intelligence.finetune import curate_dataset

    stats = curate_dataset(ROOT / args.traces, ROOT / args.out)
    print(f"Curated {stats['examples_out']} examples from {stats['traces_in']} traces "
          f"({stats['dropped']} dropped) -> {stats['out_path']}")

    # NOTE: the Trainer wiring (tokenizer, data collator, LoraConfig from
    # the YAML) is intentionally left as the next step for the GPU run:
    # the configs in configs/ are complete and the dataset format is the
    # standard {"instruction","input","output"} SFT shape, so plugging in
    # trl.SFTTrainer is a few lines. Training itself is out of scope for
    # the CPU demo and is not faked here.
    print("Config:", args.config)
    print("Next: wire trl.SFTTrainer with the LoRA config and run on GPU. "
          "See README 'Fine-tuning on real traces'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
