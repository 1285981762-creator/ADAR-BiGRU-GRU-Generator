from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .inference import BiGRUGRUGenerator, Target
from .ops import normalize_rna, reverse_complement


DEFAULT_MODEL = Path(__file__).resolve().parent / "weights" / "bigru_gru_v3.pt"


def read_targets(path: Path) -> list[Target]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "target_sequence" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a target_sequence column")
        targets = []
        for row_number, row in enumerate(reader, start=1):
            sequence = normalize_rna(row.get("target_sequence", ""))
            target_id = (row.get("target_id") or f"target_{row_number}").strip()
            if sequence.count("V") != 1 or set(sequence) - set("AUGCV"):
                raise ValueError(
                    f"Row {row_number}: target_sequence must contain exactly one V "
                    "and only A/U/G/C/V"
                )
            targets.append(Target(target_id=target_id, sequence=sequence))
    if not targets:
        raise ValueError("Input CSV contains no targets")
    return targets


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate ADAR complementary RNA candidates with BiGRU-GRU"
    )
    result.add_argument("--input", type=Path, required=True, help="Target CSV")
    result.add_argument("--output", type=Path, required=True, help="Output candidate CSV")
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--num-sequences", type=int, default=4, help="Rows per target")
    result.add_argument("--efficiency", type=float, default=80.0, help="Desired efficiency, 0-100")
    result.add_argument("--percentile", type=float, default=0.90, help="Desired within-site percentile, 0-1")
    result.add_argument(
        "--min-complementarity",
        type=float,
        default=None,
        help="Requested floor, 0-1; default reads 0.90 from the checkpoint",
    )
    result.add_argument(
        "--max-edits", type=int, default=None, help="Optional stricter edit-action cap"
    )
    result.add_argument("--guidance-scale", type=float, default=2.0)
    result.add_argument("--temperature", type=float, default=0.8)
    result.add_argument("--top-k", type=int, default=6)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--seed", type=int, default=20260922)
    result.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    result.add_argument(
        "--include-strict-complement",
        action="store_true",
        help="Append one 100%% complementary baseline row per target",
    )
    return result


def validate_args(args):
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be at least 1")
    if not 0 <= args.efficiency <= 100:
        raise ValueError("--efficiency must be between 0 and 100")
    if not 0 <= args.percentile <= 1:
        raise ValueError("--percentile must be between 0 and 1")
    if args.min_complementarity is not None and not 0 <= args.min_complementarity <= 1:
        raise ValueError("--min-complementarity must be between 0 and 1")
    if args.max_edits is not None and args.max_edits < 0:
        raise ValueError("--max-edits cannot be negative")
    if args.guidance_scale < 1:
        raise ValueError("--guidance-scale must be at least 1")
    if args.temperature < 0 or args.top_k < 0 or args.batch_size < 1:
        raise ValueError("Invalid sampling setting")


def main():
    args = parser().parse_args()
    try:
        validate_args(args)
        targets = read_targets(args.input)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        generator = BiGRUGRUGenerator(args.model, args.device)
        floor = (
            generator.training_complementarity_floor
            if args.min_complementarity is None
            else args.min_complementarity
        )
        if floor < generator.training_complementarity_floor:
            print(
                f"Warning: requested complementarity {floor:.3f} is below the "
                f"training floor {generator.training_complementarity_floor:.3f}.",
                file=sys.stderr,
            )
        rows = []
        for target in targets:
            rows.extend(
                generator.generate_target(
                    target=target,
                    num_sequences=args.num_sequences,
                    efficiency=args.efficiency,
                    percentile=args.percentile,
                    min_complementarity=floor,
                    max_edits=args.max_edits,
                    guidance_scale=args.guidance_scale,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                )
            )
            if args.include_strict_complement:
                strict = reverse_complement(target.sequence)
                rows.append(
                    {
                        "candidate_id": f"{target.target_id}_strict_complement",
                        "target_id": target.target_id,
                        "target_sequence": normalize_rna(target.sequence),
                        "edit_position_1based": normalize_rna(target.sequence).index("V") + 1,
                        "complement_sequence_5to3": strict,
                        "desired_efficiency_percent": "",
                        "desired_within_site_percentile": "",
                        "requested_min_complementarity": 1.0,
                        "actual_complementarity": 1.0,
                        "edit_distance_from_strict_complement": 0,
                        "edit_action_count": 0,
                        "internal_quality_score": "",
                        "sample_number": 0,
                    }
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "model": str(args.model),
            "model_type": generator.model_type,
            "device": str(generator.device),
            "targets": len(targets),
            "generated_rows": len(targets) * args.num_sequences,
            "strict_complement_rows": len(targets) if args.include_strict_complement else 0,
            "desired_efficiency_percent": args.efficiency,
            "desired_within_site_percentile": args.percentile,
            "minimum_complementarity": floor,
            "output": str(args.output),
            "note": "Desired efficiency is a generation condition, not a calibrated wet-lab prediction.",
        }
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
