from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

from adar_bigru.ops import encode_source_and_template, reverse_complement


EXPECTED_SHA256 = "3865D4F887511FC0472FD1E44AE522BD87FDC9186BB6F25FD2F4FB57EEAB7EA7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    target = "AUGCVACGU"
    _, template, _ = encode_source_and_template(target)
    assert template == reverse_complement(target)
    weight = Path(__file__).resolve().parents[1] / "adar_bigru" / "weights" / "bigru_gru_v3.pt"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest().upper()
    assert digest == EXPECTED_SHA256
    report = {"ops_test": True, "weight_sha256_match": True}
    if args.wheel:
        with ZipFile(args.wheel) as archive:
            names = set(archive.namelist())
        required = {
            "adar_bigru/cli.py",
            "adar_bigru/inference.py",
            "adar_bigru/weights/bigru_gru_v3.pt",
        }
        report["wheel_required_files_present"] = required <= names
        assert report["wheel_required_files_present"]
    print(report)


if __name__ == "__main__":
    main()
