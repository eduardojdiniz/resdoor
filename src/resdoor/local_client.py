"""Local PyTorch inference client implementing :class:`~resdoor.models.ProbeClient`.

Uses HuggingFace ``transformers`` to load the warmup model and extract
activations via ``register_forward_hook``.  All synchronous PyTorch calls
are wrapped with :func:`asyncio.to_thread` so the async Protocol contract
is satisfied.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from resdoor.models import Hypothesis, prompt_hash

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "jane-street/dormant-model-warmup"

# Default module names targeting first, middle, and last layers of a
# 32-layer Qwen2 / Llama-3 8B architecture.
_DEFAULT_MODULE_NAMES = (
    "model.layers.0.mlp.down_proj",
    "model.layers.15.mlp.down_proj",
    "model.layers.31.mlp.down_proj",
)


class LocalClient:
    """Local PyTorch inference backend implementing ``ProbeClient``.

    Downloads and loads the warmup model on first call (lazy initialisation).
    Inference uses ``model.generate()`` for chat and
    ``register_forward_hook()`` for activation extraction.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        PyTorch device (e.g. ``"cuda"``, ``"cpu"``, ``"mps"``).
    dtype : torch.dtype
        Model dtype.  Defaults to ``torch.bfloat16``.
    max_new_tokens : int
        Maximum tokens to generate per prompt.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._max_new_tokens = max_new_tokens
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def _ensure_loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Lazily load the model and tokenizer on first use."""
        if self._model is None or self._tokenizer is None:
            _log.info("Loading model %s on %s (%s)", self._model_name, self._device, self._dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=self._dtype,
                device_map=self._device,
            )
            self._model.eval()
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Synchronous inference helpers (run on thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _generate_text(self, prompt: str) -> str:
        """Generate a response for a single prompt."""
        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens (skip the input)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True))

    def _extract_activations(self, prompt: str, module_names: list[str]) -> np.ndarray:
        """Extract mean-pooled activation vectors for specified modules."""
        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        captured: dict[str, np.ndarray] = {}
        hooks = []

        def _make_hook(name: str):
            def hook_fn(module, input, output):
                # output is typically a tensor of shape (batch, seq_len, hidden_dim)
                tensor = output
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                # Mean-pool over token dimension -> (hidden_dim,)
                captured[name] = tensor[0].mean(dim=0).detach().float().cpu().numpy()

            return hook_fn

        # Register hooks
        for name in module_names:
            parts = name.split(".")
            target = model
            for part in parts:
                target = getattr(target, part)
            hooks.append(target.register_forward_hook(_make_hook(name)))

        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Concatenate in module_names order -> 1-D array
        vectors = [captured[name] for name in module_names if name in captured]
        if not vectors:
            return np.array([], dtype=np.float32)
        return np.concatenate(vectors)

    def _sync_fetch_baselines(
        self,
        model: str,
        prompts: tuple[str, ...],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch baselines synchronously via local model."""
        chat_baselines: dict[str, str] = {}
        act_baselines: dict[str, np.ndarray] = {}

        for prompt in prompts:
            _log.info("Local baseline: %s (model=%s)", prompt[:50], model)
            chat_baselines[prompt] = self._generate_text(prompt)
            if module_names:
                act_baselines[prompt] = self._extract_activations(prompt, module_names)

        return chat_baselines, act_baselines

    def _sync_fetch_triggered(
        self,
        model: str,
        hypothesis_prompts: list[tuple[Hypothesis, str, str]],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch triggered responses synchronously via local model."""
        chat_triggered: dict[str, str] = {}
        act_triggered: dict[str, np.ndarray] = {}

        for hyp, base_prompt, triggered_prompt in hypothesis_prompts:
            key = f"{hyp.id}|{prompt_hash(base_prompt)}"
            _log.info("Local triggered: %s (hyp=%s)", triggered_prompt[:50], hyp.id)
            chat_triggered[key] = self._generate_text(triggered_prompt)
            if module_names:
                act_triggered[key] = self._extract_activations(triggered_prompt, module_names)

        return chat_triggered, act_triggered

    # ------------------------------------------------------------------
    # ProbeClient Protocol methods (async, wrapping sync via to_thread)
    # ------------------------------------------------------------------

    async def fetch_baselines(
        self,
        model: str,
        prompts: tuple[str, ...],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch baseline chat text and activation vectors for all prompts.

        Parameters
        ----------
        model : str
            Model identifier (used for logging; actual model is set at init).
        prompts : tuple[str, ...]
            Baseline prompts.
        module_names : list[str]
            Activation module names.

        Returns
        -------
        tuple[dict[str, str], dict[str, np.ndarray]]
            ``(chat_baselines, activation_baselines)`` keyed by prompt.
        """
        return await asyncio.to_thread(self._sync_fetch_baselines, model, prompts, module_names)

    async def fetch_triggered(
        self,
        model: str,
        hypothesis_prompts: list[tuple[Hypothesis, str, str]],
        module_names: list[str],
    ) -> tuple[dict[str, str], dict[str, np.ndarray]]:
        """Fetch triggered chat text and activation vectors for hypotheses.

        Parameters
        ----------
        model : str
            Model identifier (used for logging; actual model is set at init).
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
        return await asyncio.to_thread(self._sync_fetch_triggered, model, hypothesis_prompts, module_names)
