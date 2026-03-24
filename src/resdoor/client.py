"""High-level async client wrapping `jsinfer.BatchInferenceClient`."""

from __future__ import annotations

from collections.abc import Iterable

from jsinfer import (
    ActivationsRequest,
    ActivationsResponse,
    BatchInferenceClient,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)

MODELS = (
    "dormant-model-warmup",
    "dormant-model-1",
    "dormant-model-2",
    "dormant-model-3",
)


class ResdoorClient:
    """Thin async wrapper around :class:`jsinfer.BatchInferenceClient`."""

    def __init__(self) -> None:
        self._client = BatchInferenceClient()

    async def request_access(self, email: str) -> None:
        """Request an API key to be sent to *email*."""
        await self._client.request_access(email)

    def set_api_key(self, api_key: str) -> None:
        """Set the API key obtained from the activation email."""
        self._client.set_api_key(api_key)

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def chat(
        self,
        prompts: Iterable[str],
        model: str,
        *,
        system: str | None = None,
    ) -> dict[str, ChatCompletionResponse]:
        """Send *prompts* to *model* and return completion results."""
        requests = []
        for i, prompt in enumerate(prompts):
            messages: list[Message] = []
            if system is not None:
                messages.append(Message(role="system", content=system))
            messages.append(Message(role="user", content=prompt))
            requests.append(
                ChatCompletionRequest(
                    custom_id=f"entry-{i:04d}",
                    messages=messages,
                )
            )
        return await self._client.chat_completions(requests, model=model)

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    async def activations(
        self,
        prompts: Iterable[str],
        model: str,
        *,
        layers: list[int] | None = None,
    ) -> dict[str, ActivationsResponse]:
        """Retrieve internal activations for *prompts* from *model*."""
        requests = []
        for i, prompt in enumerate(prompts):
            kwargs: dict = {}
            if layers is not None:
                kwargs["layers"] = layers
            requests.append(
                ActivationsRequest(
                    custom_id=f"entry-{i:04d}",
                    messages=[Message(role="user", content=prompt)],
                    **kwargs,
                )
            )
        return await self._client.activations(requests, model=model)
