from __future__ import annotations

from typing import Any

import tinker
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.models.metric import UsageInfo
from pydantic import BaseModel, ConfigDict, PrivateAttr
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class TinkerLLM(BaseLLM, BaseModel):
    """LLM backend using Tinker SamplingClient for Harbor agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    context_limit: int = 32000

    _client: tinker.SamplingClient = PrivateAttr()
    _tokenizer: Tokenizer = PrivateAttr()
    _renderer: Renderer = PrivateAttr()

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        tokenizer: Tokenizer,
        renderer: Renderer,
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        context_limit: int = 32000,
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            context_limit=context_limit,
        )
        self._client = sampling_client
        self._tokenizer = tokenizer
        self._renderer = renderer

    def get_model_context_limit(self) -> int:
        return self.context_limit

    def update_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Update the sampling client after weight updates."""
        self._client = sampling_client

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Sample from Tinker and return an LLMResponse."""
        messages = (message_history or []) + [{"role": "user", "content": prompt}]
        model_input = self._renderer.build_generation_prompt(messages)

        if model_input.length > self.context_limit - self.max_tokens:
            raise ContextLengthExceededError(
                f"Context length {model_input.length} exceeds limit {self.context_limit - self.max_tokens}"
            )

        result = await self._client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self._renderer.get_stop_sequences(),
            ),
        )

        seq = result.sequences[0]
        tokens = list(seq.tokens)
        logprobs = list(seq.logprobs) if seq.logprobs else None

        if logprobs is None:
            raise RuntimeError("Tinker response missing logprobs (required for RL training)")

        content = self._tokenizer.decode(tokens)

        if seq.stop_reason == "length":
            raise OutputLengthExceededError(
                f"Output truncated at {self.max_tokens} tokens",
                truncated_response=content,
            )

        return LLMResponse(
            content=content,
            reasoning_content=None,
            usage=UsageInfo(
                prompt_tokens=model_input.length,
                completion_tokens=len(tokens),
                cache_tokens=0,
                cost_usd=0.0,
            ),
            prompt_token_ids=model_input.to_ints(),
            completion_token_ids=tokens,
            logprobs=logprobs,
        )
