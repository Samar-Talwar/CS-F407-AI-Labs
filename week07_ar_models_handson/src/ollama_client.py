# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Ollama Client Interface & Wrapper.

Provides:
- Health check for local/remote Ollama instance
- Listing of available models
- Text generation and chat endpoints
- Deterministic offline stub mode with exact testable responses
- Prompt template and chaining interface mirroring `Run_Ollama.ipynb`
- Request builder and response parser for REST API integration
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaResponse:
    """Standardized response from Ollama (stub or real)."""

    text: str
    model: str
    backend: str  # "stub" or "real"
    done: bool
    status_code: int
    raw_response: dict[str, Any] | None = None


class PromptTemplate:
    """Simple prompt template matching LangChain style templates."""

    def __init__(self, template: str) -> None:
        self.template = template

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        return cls(template)

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def __or__(self, other: Any) -> OllamaChain:
        if isinstance(other, (OllamaClient, OllamaLLMStub)):
            return OllamaChain(template=self, client=other)
        msg = f"Unsupported operand type for |: {type(other)}"
        raise TypeError(msg)


class OllamaChain:
    """Prompt-model pipeline chain matching `prompt | model` syntax in Run_Ollama.ipynb."""

    def __init__(self, template: PromptTemplate, client: OllamaClient | OllamaLLMStub) -> None:
        self.template = template
        self.client = client

    def invoke(self, inputs: dict[str, Any]) -> str:
        prompt = self.template.format(**inputs)
        response = self.client.generate(prompt=prompt)
        return response.text


class OllamaClient:
    """
    Client wrapper for Ollama REST API with deterministic offline stub.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 2.0,
        force_stub: bool = False,
        default_model: str = "mistral",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.force_stub = force_stub
        self.default_model = default_model

    def health_check(self) -> bool:
        """
        Check if the Ollama server is running and accessible.
        """
        if self.force_stub:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        Return list of model names installed on the Ollama server, or stub fallback list.
        """
        if self.force_stub or not self.health_check():
            return ["mistral", "llama3", "qwen2.5", "qwen2.5:3b"]

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return [m for m in models if m]
        except Exception:
            return ["mistral", "llama3", "qwen2.5", "qwen2.5:3b"]

    def build_generate_payload(
        self,
        prompt: str,
        model: str | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build JSON payload for Ollama /api/generate endpoint."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        return payload

    def parse_response(self, raw_data: dict[str, Any], backend: str = "real") -> OllamaResponse:
        """Parse raw JSON dict from Ollama API into OllamaResponse."""
        text = raw_data.get("response", "")
        model = raw_data.get("model", self.default_model)
        done = raw_data.get("done", True)
        return OllamaResponse(
            text=text,
            model=model,
            backend=backend,
            done=done,
            status_code=200,
            raw_response=raw_data,
        )

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> OllamaResponse:
        """
        Generate text completion from Ollama (or offline deterministic stub if server unreachable).
        """
        target_model = model or self.default_model
        full_prompt = f"System: {system}\n\n{prompt}" if system else prompt

        if self.force_stub or not self.health_check():
            stub_text = self._get_stub_generation(full_prompt, target_model)
            return OllamaResponse(
                text=stub_text,
                model=target_model,
                backend="stub",
                done=True,
                status_code=200,
                raw_response={"response": stub_text, "model": target_model, "done": True},
            )

        # Real backend HTTP call
        payload = self.build_generate_payload(
            prompt=full_prompt, model=target_model, options=options
        )
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return self.parse_response(resp_data, backend="real")
        except urllib.error.URLError as e:
            # Fallback gracefully with clear error context
            stub_text = (
                f"[Ollama server at {self.base_url} unavailable: {e.reason}. Using stub]\n"
                + self._get_stub_generation(full_prompt, target_model)
            )
            return OllamaResponse(
                text=stub_text,
                model=target_model,
                backend="stub",
                done=True,
                status_code=503,
                raw_response={"error": str(e.reason)},
            )

    def _get_stub_generation(self, prompt: str, model: str) -> str:
        """Deterministic, grounded responses for standard lab prompts."""
        p_lower = prompt.lower()
        if "newton" in p_lower:
            return (
                "Sir Isaac Newton (1642-1727): English mathematician, physicist, astronomer. "
                "Formulated laws of motion/gravitation; invented calculus; "
                "built reflecting telescope."
            )
        if "plants create energy" in p_lower or "photosynthesis" in p_lower:
            return "Photosynthesis, converting light energy into chemical energy stored in glucose."
        if "future of ai" in p_lower:
            return (
                "Future of AI: neurosymbolic reasoning + autonomous agents "
                "+ verified generative systems."
            )
        return (
            f"[Stub response from {model}]: Received prompt with {len(prompt.split())} words. "
            "Thinking step by step to solve the query."
        )


class OllamaLLMStub(OllamaClient):
    """Convenience alias for LangChain compatibility."""

    def __init__(self, model: str = "mistral", **kwargs: Any) -> None:
        super().__init__(default_model=model, **kwargs)


def run_newton_query(client: OllamaClient | None = None) -> dict[str, Any]:
    """
    Execute the exact prompt chain from Run_Ollama.ipynb:
    template = "Question: {question}\\n\\nAnswer: Let's think step by step."
    chain = prompt | model
    chain.invoke({"question": "Who is Sir Issac Newton"})
    """
    if client is None:
        client = OllamaClient(force_stub=True)

    template = PromptTemplate.from_template(
        "Question: {question}\n\nAnswer: Let's think step by step."
    )
    chain = template | client
    query = "Who is Sir Issac Newton"
    answer = chain.invoke({"question": query})

    return {
        "query": query,
        "template": template.template,
        "model": client.default_model,
        "backend": "stub" if client.force_stub or not client.health_check() else "real",
        "answer": answer,
    }
