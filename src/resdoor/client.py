"""High-level async client wrapping ``jsinfer.BatchInferenceClient``."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from aiohttp import ClientResponseError
from jsinfer import (
    ActivationsRequest,
    ActivationsResponse,
    BatchInferenceClient,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)

from resdoor.analysis import extract_activation_vectors
from resdoor.models import Hypothesis, ProbeConfig, RateLimitConfig, ResdoorSettings, prompt_hash

_log = logging.getLogger(__name__)

_DEFAULT_RATE_LIMIT = RateLimitConfig()

MODELS = (
    "dormant-model-1",
    "dormant-model-2",
    "dormant-model-3",
)


class CreditExhausted(RuntimeError):  # noqa: N818
    """Raised when the 429 retry budget is fully exhausted.

    Attributes
    ----------
    batch_id : str
        The batch identifier that triggered exhaustion.
    retries : int
        Total number of retries attempted before giving up.
    """


class ResdoorClient:
    """Thin async wrapper around :class:`jsinfer.BatchInferenceClient`.

    Parameters
    ----------
    settings : ResdoorSettings
        Application settings containing the API key and default models.
    rate_limit : RateLimitConfig
        Polling and backoff configuration.  Uses sensible defaults when
        omitted.
    """

    def __init__(
        self,
        settings: ResdoorSettings,
        rate_limit: RateLimitConfig = _DEFAULT_RATE_LIMIT,
    ) -> None:
        self._settings = settings
        self.rate_limit = rate_limit
        self._client = BatchInferenceClient(api_key=settings.api_key)
        self._client.poll_batch = self._rate_limited_poll_batch  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Rate-limited polling
    # ------------------------------------------------------------------

    async def _rate_limited_poll_batch(
        self,
        batch_id: str,
        timeout: int = 86400,
    ) -> str:
        """Poll a batch job with exponential backoff on HTTP 429.

        Parameters
        ----------
        batch_id : str
            Identifier of the batch to poll.
        timeout : int
            Maximum wall-clock seconds before raising a timeout error.

        Returns
        -------
        str
            The ``resultsUrl`` of the completed batch.

        Raises
        ------
        CreditExhausted
            If the 429 retry budget is fully exhausted.
        RuntimeError
            If the batch enters a terminal failure state or *timeout*
            is exceeded.
        """
        interval = self.rate_limit.poll_interval
        retries_left = self.rate_limit.max_poll_retries
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                batch = await self._client.get_batch(batch_id)
            except ClientResponseError as exc:
                if exc.status == 429:
                    retries_left -= 1
                    if retries_left <= 0:
                        raise CreditExhausted(
                            f"Batch {batch_id}: poll retry budget exhausted "
                            f"after {self.rate_limit.max_poll_retries} 429 responses"
                        ) from exc
                    interval = min(
                        interval * self.rate_limit.poll_backoff_factor,
                        self.rate_limit.poll_max_backoff,
                    )
                    jitter = random.uniform(0, self.rate_limit.poll_jitter)
                    _log.warning(
                        "Batch %s: 429 rate-limited, backing off %.1fs (+%.2fs jitter)",
                        batch_id,
                        interval,
                        jitter,
                    )
                    await asyncio.sleep(interval + jitter)
                    continue
                raise

            status: str = batch["batch"]["status"]
            match status:
                case "completed":
                    return batch["resultsUrl"]  # type: ignore[no-any-return]
                case "failed" | "cancelled" | "expired" | "error":
                    raise RuntimeError(f"Batch {batch_id} terminal status: {status}")

            await asyncio.sleep(interval)

        raise RuntimeError(f"Batch {batch_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def chat(
        self,
        prompts: Iterable[str],
        model: str,
        *,
        system: str | None = None,
    ) -> dict[str, ChatCompletionResponse]:
        """Send *prompts* to *model* and return completion results.

        Parameters
        ----------
        prompts : Iterable[str]
            User-turn strings to send.
        model : str
            Target model identifier.
        system : str | None
            Optional system prompt prepended to every request.
        """
        requests: list[ChatCompletionRequest] = []
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
        return await self._client.chat_completions(requests, model=model)  # type: ignore[no-any-return]

    async def activations(
        self,
        prompts: Iterable[str],
        model: str,
        *,
        module_names: list[str],
    ) -> dict[str, ActivationsResponse]:
        """Retrieve internal activations for *prompts* from *model*.

        Parameters
        ----------
        prompts : Iterable[str]
            User-turn strings to send.
        model : str
            Target model identifier.
        module_names : list[str]
            Model module names whose activations should be returned.
        """
        requests: list[ActivationsRequest] = []
        for i, prompt in enumerate(prompts):
            requests.append(
                ActivationsRequest(
                    custom_id=f"entry-{i:04d}",
                    messages=[Message(role="user", content=prompt)],
                    module_names=module_names,
                )
            )
        return await self._client.activations(requests, model=model)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Probe helpers (ProbeConfig → jsinfer)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chat_requests(configs: tuple[ProbeConfig, ...]) -> dict[str, list[ChatCompletionRequest]]:
        """Convert probe configs into per-model chat completion requests.

        For each config, baseline prompts use ``custom_id = "baseline-{model}-{idx:04d}"``
        and triggered prompts prepend the hypothesis trigger with
        ``custom_id = "{hypothesis.id}-{model}-{idx:04d}"``.

        Parameters
        ----------
        configs : tuple[ProbeConfig, ...]
            Probe configurations to convert.

        Returns
        -------
        dict[str, list[ChatCompletionRequest]]
            Mapping of model name to the list of requests for that model.
        """
        by_model: dict[str, list[ChatCompletionRequest]] = {}
        for cfg in configs:
            model = cfg.model
            if model not in by_model:
                by_model[model] = []
            for idx, prompt in enumerate(cfg.base_prompts):
                # Baseline request (no trigger)
                by_model[model].append(
                    ChatCompletionRequest(
                        custom_id=f"baseline-{model}-{idx:04d}",
                        messages=[Message(role="user", content=prompt)],
                    )
                )
                # Triggered request
                triggered_prompt = f"{cfg.hypothesis.trigger} {prompt}"
                by_model[model].append(
                    ChatCompletionRequest(
                        custom_id=f"{cfg.hypothesis.id}-{model}-{idx:04d}",
                        messages=[Message(role="user", content=triggered_prompt)],
                    )
                )
        return by_model

    @staticmethod
    def _build_activations_requests(
        configs: tuple[ProbeConfig, ...],
        module_names: tuple[str, ...],
    ) -> dict[str, list[ActivationsRequest]]:
        """Convert probe configs into per-model activations requests.

        Parameters
        ----------
        configs : tuple[ProbeConfig, ...]
            Probe configurations to convert.
        module_names : tuple[str, ...]
            Model module names whose activations should be returned.

        Returns
        -------
        dict[str, list[ActivationsRequest]]
            Mapping of model name to the list of requests for that model.
        """
        by_model: dict[str, list[ActivationsRequest]] = {}
        for cfg in configs:
            model = cfg.model
            if model not in by_model:
                by_model[model] = []
            for idx, prompt in enumerate(cfg.base_prompts):
                # Baseline request (no trigger)
                by_model[model].append(
                    ActivationsRequest(
                        custom_id=f"baseline-{model}-{idx:04d}",
                        messages=[Message(role="user", content=prompt)],
                        module_names=list(module_names),
                    )
                )
                # Triggered request
                triggered_prompt = f"{cfg.hypothesis.trigger} {prompt}"
                by_model[model].append(
                    ActivationsRequest(
                        custom_id=f"{cfg.hypothesis.id}-{model}-{idx:04d}",
                        messages=[Message(role="user", content=triggered_prompt)],
                        module_names=list(module_names),
                    )
                )
        return by_model

    async def probe_chat(
        self,
        configs: tuple[ProbeConfig, ...],
    ) -> dict[str, ChatCompletionResponse]:
        """Run chat-completion probes for baseline and triggered prompts.

        For each :class:`ProbeConfig`, sends both baseline (unmodified) and
        triggered (hypothesis trigger prepended) prompts and returns all
        responses keyed by ``custom_id``.

        Parameters
        ----------
        configs : tuple[ProbeConfig, ...]
            Probe configurations describing hypotheses, models, and prompts.

        Returns
        -------
        dict[str, ChatCompletionResponse]
            All responses keyed by ``custom_id``.
        """
        by_model = self._build_chat_requests(configs)
        results: dict[str, ChatCompletionResponse] = {}
        for model, requests in by_model.items():
            batch = await self._client.chat_completions(requests, model=model)
            results.update(batch)
        return results

    async def probe_activations(
        self,
        configs: tuple[ProbeConfig, ...],
        module_names: tuple[str, ...],
    ) -> dict[str, ActivationsResponse]:
        """Run activation probes for baseline and triggered prompts.

        For each :class:`ProbeConfig`, sends both baseline (unmodified) and
        triggered (hypothesis trigger prepended) prompts and returns all
        activation responses keyed by ``custom_id``.

        Parameters
        ----------
        configs : tuple[ProbeConfig, ...]
            Probe configurations describing hypotheses, models, and prompts.
        module_names : tuple[str, ...]
            Model module names whose activations should be returned.

        Returns
        -------
        dict[str, ActivationsResponse]
            All responses keyed by ``custom_id``.
        """
        by_model = self._build_activations_requests(configs, module_names)
        results: dict[str, ActivationsResponse] = {}
        for model, requests in by_model.items():
            batch = await self._client.activations(requests, model=model)
            results.update(batch)
        return results

    # ------------------------------------------------------------------
    # Baseline caching
    # ------------------------------------------------------------------

    _BASELINES_DIR = Path("data/baselines")

    @staticmethod
    def _baseline_cache_path(model: str, prompt: str, kind: str) -> Path:
        """Return the cache file path for a baseline response."""
        return ResdoorClient._BASELINES_DIR / f"{model}_{prompt_hash(prompt)}_{kind}.json"

    @staticmethod
    def _load_cached_chat(model: str, prompt: str) -> str | None:
        """Load cached baseline chat text, or ``None`` if not cached."""
        path = ResdoorClient._baseline_cache_path(model, prompt, "chat")
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data.get("text")  # type: ignore[no-any-return]

    @staticmethod
    def _save_chat_cache(model: str, prompt: str, text: str) -> None:
        """Persist baseline chat text to cache."""
        ResdoorClient._BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        path = ResdoorClient._baseline_cache_path(model, prompt, "chat")
        path.write_text(json.dumps({"text": text}))

    @staticmethod
    def _load_cached_activations(model: str, prompt: str) -> np.ndarray | None:
        """Load cached baseline activation vectors, or ``None`` if not cached."""
        npy_path = ResdoorClient._baseline_cache_path(model, prompt, "activations").with_suffix(".npy")
        if not npy_path.exists():
            return None
        return np.load(npy_path)

    @staticmethod
    def _save_activations_cache(model: str, prompt: str, vecs: np.ndarray) -> None:
        """Persist baseline activation vectors to cache."""
        ResdoorClient._BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        npy_path = ResdoorClient._baseline_cache_path(model, prompt, "activations").with_suffix(".npy")
        np.save(npy_path, vecs)

    @staticmethod
    def _extract_chat_text(chat_results: dict[str, ChatCompletionResponse], key: str) -> str:
        """Extract the assistant reply text from a chat completion result."""
        resp = chat_results.get(key)
        if resp is None:
            return ""
        for msg in resp.messages:
            if msg.role == "assistant":
                return str(msg.content)
        return ""

    # ------------------------------------------------------------------
    # ProbeClient Protocol methods
    # ------------------------------------------------------------------

    async def fetch_baselines(
        self,
        model: str,
        prompts: tuple[str, ...],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch baseline chat + activations for all prompts, using cache.

        Batches all un-cached prompts into single API calls per endpoint.

        Parameters
        ----------
        model : str
            Model identifier.
        prompts : tuple[str, ...]
            Baseline prompts.
        module_names : list[str]
            Activation module names.

        Returns
        -------
        tuple[dict[str, str], dict[str, np.ndarray]]
            ``(chat_baselines, activation_baselines)`` keyed by prompt.
        """
        chat_baselines: dict[str, str] = {}
        act_baselines: dict[str, np.ndarray] = {}
        uncached_chat: list[str] = []
        uncached_act: list[str] = []

        for prompt in prompts:
            cached_chat = self._load_cached_chat(model, prompt)
            if cached_chat is not None:
                chat_baselines[prompt] = cached_chat
            else:
                uncached_chat.append(prompt)

            cached_act = self._load_cached_activations(model, prompt)
            if cached_act is not None:
                act_baselines[prompt] = cached_act
            else:
                uncached_act.append(prompt)

        if uncached_chat:
            _log.info("Fetching %d baseline chats for %s", len(uncached_chat), model)
            results = await self.chat(uncached_chat, model)
            for i, prompt in enumerate(uncached_chat):
                text = self._extract_chat_text(results, f"entry-{i:04d}")
                chat_baselines[prompt] = text
                self._save_chat_cache(model, prompt, text)

        if uncached_act and module_names:
            _log.info("Fetching %d baseline activations for %s", len(uncached_act), model)
            results = await self.activations(uncached_act, model, module_names=module_names)
            for i, prompt in enumerate(uncached_act):
                resp = results.get(f"entry-{i:04d}")
                if resp is not None:
                    vecs = extract_activation_vectors(resp.activations)
                    act_baselines[prompt] = vecs
                    self._save_activations_cache(model, prompt, vecs)

        return chat_baselines, act_baselines

    async def fetch_triggered(
        self,
        model: str,
        hypothesis_prompts: list[tuple[Hypothesis, str, str]],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Batch all triggered requests for a model into API calls.

        Parameters
        ----------
        model : str
            Model identifier.
        hypothesis_prompts : list[tuple[Hypothesis, str, str]]
            List of ``(hypothesis, base_prompt, triggered_prompt)`` tuples.
        module_names : list[str]
            Activation module names.

        Returns
        -------
        tuple[dict[str, str], dict[str, np.ndarray]]
            ``(chat_triggered, act_triggered)`` keyed by
            ``"{hyp_id}|{prompt_hash}"``.
        """
        triggered_prompts = [tp for _, _, tp in hypothesis_prompts]
        id_keys = [f"{h.id}|{prompt_hash(bp)}" for h, bp, _ in hypothesis_prompts]

        # Chat
        _log.info("Batch chat: %d triggered prompts for %s", len(triggered_prompts), model)
        chat_results = await self.chat(triggered_prompts, model)
        chat_triggered = {key: self._extract_chat_text(chat_results, f"entry-{i:04d}") for i, key in enumerate(id_keys)}

        # Activations
        act_triggered: dict[str, np.ndarray] = {}
        if module_names:
            _log.info("Batch activations: %d triggered prompts for %s", len(triggered_prompts), model)
            act_results = await self.activations(triggered_prompts, model, module_names=module_names)
            for i, key in enumerate(id_keys):
                resp = act_results.get(f"entry-{i:04d}")
                if resp is not None:
                    act_triggered[key] = extract_activation_vectors(resp.activations)

        return chat_triggered, act_triggered
