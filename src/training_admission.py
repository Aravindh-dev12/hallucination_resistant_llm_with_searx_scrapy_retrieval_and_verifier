"""CLI and library for governed admission of external training records."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from governance import AdmissionDecision, GovernancePolicy, ResourceRecord, ResourceRole


def load_policy(path: str | Path) -> GovernancePolicy:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    admission = document.get("policy", {}).get("training_admission", {})
    licenses = admission.get("compatible_licenses", [])
    if not licenses:
        raise ValueError("policy.training_admission.compatible_licenses is required")
    return GovernancePolicy(
        compatible_licenses=frozenset(licenses),
        required_verifiers=int(admission.get("required_verifiers", 1)),
        promotion_quorum=int(
            document.get("policy", {})
            .get("checkpoint_promotion", {})
            .get("independent_quorum", 2)
        ),
        require_consent=bool(
            document.get("policy", {})
            .get("checkpoint_promotion", {})
            .get("require_consent", True)
        ),
    )


def load_manifest(path: str | Path) -> list[ResourceRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    records: list[ResourceRecord] = []
    for raw in document.get("resources", []):
        version = str(raw.get("version", "")).strip()
        if not version or version.lower() == "latest":
            raise ValueError(f"{raw.get('name', '<unnamed>')}: version must be pinned")
        records.append(
            ResourceRecord(
                name=str(raw["name"]),
                role=ResourceRole(str(raw["role"])),
                version=version,
                source_uri=str(raw["source_uri"]),
                license_id=str(raw.get("license_id", "UNKNOWN")),
                content_digest=str(raw.get("content_digest", "")),
                provenance_accepted=bool(raw.get("provenance_accepted", False)),
                verifier_receipts=tuple(raw.get("verifier_receipts", [])),
                capabilities=tuple(raw.get("capabilities", [])),
            )
        )
    if not records:
        raise ValueError("manifest must contain at least one resource")
    return records


def assess_manifest(
    policy: GovernancePolicy, records: list[ResourceRecord]
) -> dict[str, Any]:
    receipts = [policy.assess_training_resource(record) for record in records]
    accepted = all(
        receipt.decision is AdmissionDecision.ACCEPT for receipt in receipts
    )
    return {
        "decision": "accept" if accepted else "reject",
        "receipts": [
            {
                **asdict(receipt),
                "decision": receipt.decision.value,
                "reasons": list(receipt.reasons),
            }
            for receipt in receipts
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an external training-resource manifest"
    )
    parser.add_argument("manifest")
    parser.add_argument(
        "--policy", default="config/governed_resources.yaml"
    )
    parser.add_argument("--output", help="optional JSON receipt output path")
    args = parser.parse_args()

    result = assess_manifest(load_policy(args.policy), load_manifest(args.manifest))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if result["decision"] == "accept" else 2


if __name__ == "__main__":
    raise SystemExit(main())
