from pathlib import Path

import pytest
import yaml

from training_admission import assess_manifest, load_manifest, load_policy


def write_yaml(path: Path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def policy_document():
    return {
        "policy": {
            "training_admission": {
                "compatible_licenses": ["MIT", "Apache-2.0"],
                "required_verifiers": 1,
            },
            "checkpoint_promotion": {
                "independent_quorum": 2,
                "require_consent": True,
            },
        }
    }


def accepted_manifest():
    return {
        "resources": [
            {
                "name": "VibeThinker",
                "role": "training_resource",
                "version": "3b@sha256:weights",
                "source_uri": "https://github.com/WeiboAI/VibeThinker",
                "license_id": "MIT",
                "content_digest": "sha256:records",
                "provenance_accepted": True,
                "verifier_receipts": ["verifier:one"],
            }
        ]
    }


def test_manifest_is_admitted_only_after_all_gates(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    write_yaml(policy_path, policy_document())
    write_yaml(manifest_path, accepted_manifest())

    result = assess_manifest(
        load_policy(policy_path), load_manifest(manifest_path)
    )
    assert result["decision"] == "accept"
    assert result["receipts"][0]["digest"]


def test_latest_is_not_a_pinned_version(tmp_path):
    manifest = accepted_manifest()
    manifest["resources"][0]["version"] = "latest"
    path = tmp_path / "manifest.yaml"
    write_yaml(path, manifest)

    with pytest.raises(ValueError, match="must be pinned"):
        load_manifest(path)


def test_unknown_license_is_rejected(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = accepted_manifest()
    manifest["resources"][0]["license_id"] = "UNKNOWN"
    write_yaml(policy_path, policy_document())
    write_yaml(manifest_path, manifest)

    result = assess_manifest(
        load_policy(policy_path), load_manifest(manifest_path)
    )
    assert result["decision"] == "reject"
