from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein
from torch.nn.utils.rnn import pad_sequence

from .architecture_models import build_model
from .model import ModelConfig
from .ops import EOS, PAD, apply_actions, encode_source_and_template, normalize_rna


@dataclass(frozen=True)
class Target:
    target_id: str
    sequence: str


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


class BiGRUGRUGenerator:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.device = resolve_device(device)
        package = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        config = ModelConfig(**package["model_config"])
        if config.architecture != "bigru_gru":
            raise ValueError(
                f"Expected a bigru_gru checkpoint, got {config.architecture!r}"
            )
        self.model = build_model(config).to(self.device)
        self.model.load_state_dict(package["model_state_dict"])
        self.model.eval()
        self.config = config
        self.training_complementarity_floor = float(
            package.get("min_complementarity", 0.90)
        )
        self.model_type = package.get("model_type", "unknown")

    def generate_target(
        self,
        target: Target,
        num_sequences: int,
        efficiency: float,
        percentile: float,
        min_complementarity: float,
        max_edits: int | None,
        guidance_scale: float,
        temperature: float,
        top_k: int,
        batch_size: int,
    ) -> list[dict]:
        sequence = normalize_rna(target.sequence)
        source_ids, template_text, template_ids = encode_source_and_template(sequence)
        if len(source_ids) > self.config.max_length:
            raise ValueError(
                f"Target {target.target_id!r} has length {len(source_ids)}, "
                f"exceeding the checkpoint limit {self.config.max_length}"
            )
        complementarity_cap = math.floor(
            (1.0 - min_complementarity) * len(source_ids) + 1e-9
        )
        effective_max_edits = complementarity_cap
        if max_edits is not None:
            effective_max_edits = min(max_edits, complementarity_cap)

        output: list[dict] = []
        generated = 0
        while generated < num_sequences:
            current = min(batch_size, num_sequences - generated)
            source = torch.tensor(
                [source_ids] * current, dtype=torch.long, device=self.device
            )
            template = torch.tensor(
                [template_ids] * current, dtype=torch.long, device=self.device
            )
            scores = torch.full(
                (current,), efficiency, dtype=torch.float32, device=self.device
            )
            ranks = torch.full(
                (current,), percentile, dtype=torch.float32, device=self.device
            )
            actions, edit_counts = self.model.generate_actions(
                source,
                template,
                scores,
                ranks,
                effective_max_edits,
                temperature=temperature,
                top_k=top_k,
                guidance_scale=guidance_scale,
            )
            clean_scripts = []
            complements = []
            for raw_script in actions.tolist():
                end = raw_script.index(EOS) + 1
                script = raw_script[:end]
                clean_scripts.append(script)
                complements.append(apply_actions(template_text, script))
            action_tensor = pad_sequence(
                [torch.tensor(script, dtype=torch.long) for script in clean_scripts],
                batch_first=True,
                padding_value=PAD,
            ).to(self.device)
            with torch.inference_mode():
                qualities = self.model.predict_quality(source, action_tensor).float().cpu()

            for local_index, (complement, quality) in enumerate(
                zip(complements, qualities.tolist(), strict=True)
            ):
                distance = Levenshtein.distance(template_text, complement)
                similarity = 1.0 - distance / max(len(template_text), len(complement))
                if similarity + 1e-9 < min_complementarity:
                    raise RuntimeError(
                        "Constrained generation violated the requested complementarity"
                    )
                sample_number = generated + local_index + 1
                output.append(
                    {
                        "candidate_id": f"{target.target_id}_candidate_{sample_number:04d}",
                        "target_id": target.target_id,
                        "target_sequence": sequence,
                        "edit_position_1based": sequence.index("V") + 1,
                        "complement_sequence_5to3": complement,
                        "desired_efficiency_percent": efficiency,
                        "desired_within_site_percentile": percentile,
                        "requested_min_complementarity": min_complementarity,
                        "actual_complementarity": similarity,
                        "edit_distance_from_strict_complement": distance,
                        "edit_action_count": int(edit_counts[local_index].item()),
                        "internal_quality_score": quality,
                        "sample_number": sample_number,
                    }
                )
            generated += current
        return output
