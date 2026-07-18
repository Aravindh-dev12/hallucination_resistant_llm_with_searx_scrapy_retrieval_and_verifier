import pytest

from model_gateway import (
    ModelGatewayError,
    ModelRegistry,
    ModelSpec,
    OpenAICompatibleModel,
)


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": " verified output "}}]}


class Session:
    def __init__(self):
        self.request = None

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return Response()


def test_openai_compatible_adapter_uses_declared_endpoint():
    session = Session()
    spec = ModelSpec(
        name="vibe",
        provider="openai_compatible",
        model="WeiboAI/VibeThinker-3B",
        purpose="candidate_generation",
        enabled=True,
        base_url="http://localhost:8000/v1/",
    )
    output = OpenAICompatibleModel(spec, session=session).generate("solve")
    assert output == "verified output"
    assert session.request[0].endswith("/v1/chat/completions")
    assert session.request[1]["json"]["model"] == spec.model


def test_disabled_model_cannot_be_selected():
    registry = ModelRegistry(
        {
            "candidate": ModelSpec(
                name="candidate",
                provider="transformers",
                model="pinned/model",
                purpose="candidate_generation",
                enabled=False,
            )
        }
    )
    with pytest.raises(ModelGatewayError, match="disabled by policy"):
        registry.require_enabled("candidate")


def test_unknown_provider_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  unsafe:\n    provider: magic\n    model: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelGatewayError, match="unsupported provider"):
        ModelRegistry.from_yaml(str(path))
