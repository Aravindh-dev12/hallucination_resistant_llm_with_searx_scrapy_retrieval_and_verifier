"""Provider-neutral runtime model gateway.

Runtime adapters do not grant training admission or checkpoint-promotion authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import requests
import yaml


class ModelGatewayError(RuntimeError):
    pass


class TextModel(Protocol):
    def generate(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str
    purpose: str
    enabled: bool = False
    base_url: str | None = None
    api_key_env: str | None = None


class OpenAICompatibleModel:
    """Adapter for vLLM, SGLang, Ollama proxies and compatible APIs."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        session: Any = requests,
        timeout: float = 120.0,
    ):
        if not spec.base_url:
            raise ModelGatewayError(f"{spec.name}: base_url is required")
        self.spec = spec
        self.api_key = api_key
        self.session = session
        self.timeout = timeout

    def generate(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.post(
            f"{self.spec.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": self.spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelGatewayError("invalid chat-completions response") from exc
        if not isinstance(text, str) or not text.strip():
            raise ModelGatewayError("model returned an empty response")
        return text.strip()


class TransformersModel:
    """Lazy local Hugging Face adapter; weights load only when selected."""

    def __init__(self, spec: ModelSpec, *, device: str | None = None):
        self.spec = spec
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model, use_fast=True
        )
        kwargs: dict[str, Any] = {}
        if self.device is None and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model, **kwargs
        )
        if self.device:
            self._model = self._model.to(self.device)

    def generate(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class ModelRegistry:
    def __init__(self, specs: Mapping[str, ModelSpec]):
        self.specs = dict(specs)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelRegistry":
        with open(path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        specs: dict[str, ModelSpec] = {}
        for name, raw in document.get("models", {}).items():
            spec = ModelSpec(
                name=name,
                provider=str(raw["provider"]),
                model=str(raw["model"]),
                purpose=str(raw.get("purpose", "generation")),
                enabled=bool(raw.get("enabled", False)),
                base_url=raw.get("base_url"),
                api_key_env=raw.get("api_key_env"),
            )
            if spec.provider not in {"transformers", "openai_compatible"}:
                raise ModelGatewayError(
                    f"{name}: unsupported provider {spec.provider}"
                )
            specs[name] = spec
        return cls(specs)

    def require_enabled(self, name: str) -> ModelSpec:
        try:
            spec = self.specs[name]
        except KeyError as exc:
            raise ModelGatewayError(f"unknown model: {name}") from exc
        if not spec.enabled:
            raise ModelGatewayError(f"model is disabled by policy: {name}")
        return spec
