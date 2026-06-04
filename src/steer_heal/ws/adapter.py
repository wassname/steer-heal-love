"""ModulatedLoRA + ModulatedPiSSA: scalar-c-modulated rank-r adapters.

`ModulatedLoRA` (free-init A,B; W untouched):
    h     = W x
    delta = (alpha / r) * B @ A @ x       # A: r×d_in, B: d_out×r
    y     = h + c * delta                  # c=0 → exact base
Init: A ~ kaiming_uniform, B ~ N(1e-4, 1e-4); tiny nonzero B breaks +c/-c
sign symmetry at init.

`ModulatedPiSSA` (SVD-extracted; W mutated to W_res = W - U_r·S_r·Vh_r):
    h     = W_res x
    delta = U · diag(S + c · Δs) · Vh · x   # buffers U/S/Vh; trainable Δs
    y     = h + delta                        # c=0 → W·x (modulo SVD round-trip)
Init: Δs ~ N(4e-4, 4e-4) (small positive bias breaks sign symmetry).
Top-r selection driven by `selection_score`; activation-driven options
need a `calibration_activations` dict.

Forked from `weight-steering-lite/src/wsl/adapter.py`. 4-bit compatible:
hooks cast x to adapter dtype; works on bnb.Linear4bit which inherits
nn.Linear. PiSSA path requires float layers (W mutation isn't reversible
on quantized buffers); enforce that at init.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    # "all-linear" = every nn.Linear minus exclusions (PEFT default).
    # Otherwise: regex substrings matched against module names.
    targets: tuple[str, ...] = ("all-linear",)
    exclude: tuple[str, ...] = ("vision_tower", "lm_head")
    dtype: torch.dtype = torch.bfloat16
    layer_range: tuple[float, float] = (0.0, 1.0)
    """Depth band as (lo, hi) fractions; (0.2, 0.8) = middle 60% of blocks.
    Modules outside the .layers.<int>. stack (embed/norm) are kept regardless
    (filter by `exclude` instead)."""


def _match(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, name) for p in patterns)


_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _layer_idx(name: str) -> int | None:
    """Pull transformer-block index from `model.layers.<int>.…`. Returns None
    for modules outside the block stack (embed, norm, lm_head). Works for HF
    Llama/Gemma/Qwen — they all use `.layers.<int>.` naming."""
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _find_targets(model: nn.Module, cfg: LoRAConfig) -> list[tuple[str, nn.Linear]]:
    all_linear = "all-linear" in cfg.targets
    candidates = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and (all_linear or _match(name, cfg.targets))
        and not _match(name, cfg.exclude)
    ]
    lo, hi = cfg.layer_range
    if (lo, hi) != (0.0, 1.0):
        idxs = {i for i in (_layer_idx(n) for n, _ in candidates) if i is not None}
        if not idxs:
            raise RuntimeError(f"layer_range={cfg.layer_range} given but no .layers.<int>. matches")
        n_layers = max(idxs) + 1
        lo_i, hi_i = int(n_layers * lo), int(n_layers * hi)
        out = [(n, m) for n, m in candidates
               if (idx := _layer_idx(n)) is None or lo_i <= idx < hi_i]
    else:
        out = candidates
    if not out:
        raise RuntimeError(f"no targets matched {cfg.targets!r} layer_range={cfg.layer_range} "
                           f"(excluded {cfg.exclude!r})")
    return out


class ModulatedLoRA:
    """Hook-based LoRA with scalar coefficient `c`.

    Not an nn.Module: `__call__` is repurposed as a context manager for
    `with lora(model, c=...):` syntax. Params live in `self.A` / `self.B`;
    use `lora.parameters()` for the optimiser.
    """

    def __init__(self, model: nn.Module, r: int = 16, alpha: float = 32.0,
                 targets: tuple[str, ...] = ("all-linear",),
                 layer_range: tuple[float, float] = (0.0, 1.0),
                 dtype: torch.dtype = torch.bfloat16):
        self.cfg = LoRAConfig(r=r, alpha=alpha, targets=targets,
                              layer_range=layer_range, dtype=dtype)
        self._handles: list = []
        self._c: float = 0.0
        self._attached: bool = False

        device = next(model.parameters()).device
        targets_found = _find_targets(model, self.cfg)
        self.A: dict[str, nn.Parameter] = {}
        self.B: dict[str, nn.Parameter] = {}
        self._target_layers: dict[str, nn.Linear] = {}
        for name, layer in targets_found:
            d_in, d_out = layer.in_features, layer.out_features
            A = torch.empty(self.cfg.r, d_in, dtype=self.cfg.dtype, device=device)
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            B = torch.zeros(d_out, self.cfg.r, dtype=self.cfg.dtype, device=device)
            # B=0 => delta = B@A = 0, so a fresh adapter is a true no-op at init: round-0 step-0
            # student logits == base, hence barrier KL(student||orig) starts at 0 (only baked
            # history can diverge). The old normal_(mean=1e-4) perturbed EVERY all-linear layer
            # in a systematic (nonzero-mean) direction, compounding across ~200 adapters into a
            # phantom ~0.6-nat KL before any training -- it sat above tau and fired the barrier
            # against nothing real. A still gets gradient via B once B leaves zero (standard LoRA).
            self.A[name] = nn.Parameter(A)
            self.B[name] = nn.Parameter(B)
            self._target_layers[name] = layer

        for p in model.parameters():
            p.requires_grad_(False)
        n_train = sum(p.numel() for p in self.parameters())
        logger.debug(f"ModulatedLoRA: {len(targets_found)} targets, r={self.cfg.r}, "
                     f"trainable={n_train:,}")

    def parameters(self):
        for p in self.A.values():
            yield p
        for p in self.B.values():
            yield p

    def _make_hook(self, name: str):
        scale = self.cfg.alpha / self.cfg.r
        A: Float[Tensor, "r i"] = self.A[name]
        B: Float[Tensor, "o r"] = self.B[name]

        def hook(layer: nn.Linear, args, y: Tensor) -> Tensor:
            if self._c == 0.0:
                return y                              # short-circuit
            (x,) = args
            xA = x.to(A.dtype)
            h = F.linear(xA, A)                       # x @ A.T via cuBLAS
            delta = F.linear(h, B)                    # h @ B.T via cuBLAS
            return y + (self._c * scale * delta).to(y.dtype)

        return hook

    @contextmanager
    def __call__(self, model: nn.Module, c: float = 1.0):
        """`with lora(model):` -> c=+1. `with lora(model, c=...):` -> custom.

        Hooks registered on enter, removed on exit. Re-entry rejected: exit
        the outer block first.
        """
        if self._attached:
            raise RuntimeError("ModulatedLoRA already attached; exit outer `with` first")
        self._c = float(c)
        for name, layer in self._target_layers.items():
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        self._attached = True
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            self._attached = False
            self._c = 0.0

    def set_coeff(self, c: float) -> None:
        self._c = float(c)

    @property
    def c(self) -> float:
        return self._c

    # ---- save / load -------------------------------------------------------

    def save(self, path: str, extra_meta: dict[str, str] | None = None) -> None:
        from safetensors.torch import save_file
        sd = {f"A.{k.replace('.', '__')}": v.detach().cpu() for k, v in self.A.items()}
        sd.update({f"B.{k.replace('.', '__')}": v.detach().cpu() for k, v in self.B.items()})
        meta = {"r": str(self.cfg.r), "alpha": str(self.cfg.alpha),
                "targets": ",".join(self.cfg.targets),
                "layer_range": f"{self.cfg.layer_range[0]},{self.cfg.layer_range[1]}"}
        if extra_meta:
            meta.update(extra_meta)
        save_file(sd, path, metadata=meta)

    def load(self, path: str) -> None:
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
        ckpt_keys = {k[2:].replace("__", ".") for k in sd if k.startswith("A.")}
        init_keys = set(self.A.keys())
        if ckpt_keys != init_keys:
            raise RuntimeError(
                f"adapter target mismatch: checkpoint has {len(ckpt_keys)} targets, "
                f"init created {len(init_keys)}; would drop {len(ckpt_keys - init_keys)} "
                f"trained matrices and leave {len(init_keys - ckpt_keys)} init slots empty."
            )
        for k in self.A:
            kk = k.replace(".", "__")
            self.A[k].data.copy_(sd[f"A.{kk}"].to(self.A[k].device, self.A[k].dtype))
            self.B[k].data.copy_(sd[f"B.{kk}"].to(self.B[k].device, self.B[k].dtype))

    @classmethod
    def from_checkpoint(cls, model: nn.Module, path: str) -> "ModulatedLoRA":
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
        targets = tuple(meta["targets"].split(","))
        lr_meta = meta.get("layer_range", "0.0,1.0")
        lo, hi = (float(v) for v in lr_meta.split(","))
        lora = cls(model, r=int(meta["r"]), alpha=float(meta["alpha"]),
                   targets=targets, layer_range=(lo, hi),
                   dtype=next(model.parameters()).dtype)
        lora.load(path)
        return lora


# ---------------------------------------------------------------------------
# ModulatedPiSSA — top-r SVD of W is physically extracted into the adapter.
# `layer.weight` is mutated to W_res = W - U_r·S_r·Vh_r at init; the forward
# hook reconstructs U·diag(S + c·Δs)·Vh·x. At Δs=0 the SVD round-trip gives
# back W·x (bf16 noise tolerated). Δs can ablate a singular dim entirely
# (Δs_i = -S_i / c kills it) — the parameterisation gives the optimiser
# *signed authority over each top singular direction*, which is the point.
#
# Top-r selection (`selection_score`):
#   - "s_only"       : top S_full (PiSSA default — broad/blunt for instruct LMs)
#   - "wanda"        : S_full · ||X·Vh_full||  (Wanda pruning score; biases top-S)
#   - "act_only"     : ||X·Vh_full||           (pure activation magnitude)
#   - "cho_rej_min_std": min(std(X_cho·Vh_i), std(X_rej·Vh_i))
#         Picks directions that are *alive in BOTH* cho and rej completions.
#         A dim that is quiet in either mode gets a tiny min and is skipped.
#         Intuition: we want a steering axis the model actively uses
#         regardless of which side it currently leans — gives Δs a real
#         substrate at inference. Earlier "sqrt_s_act"/"wanda" picked
#         shared-fluency directions (cho and rej both use them) which
#         produced cos(g_pos, g_neg) ≈ -0.97 — pure c-sign anti-symmetry,
#         no persona signal. cho_rej_min_std needs `calibration_activations`
#         passed as `{name: {"cho": X_cho, "rej": X_rej}}`.
# Activation-scored options need `calibration_activations[name] = X (N, d_in)`.
# ---------------------------------------------------------------------------


_PISSA_SCORES = ("s_only", "wanda", "act_only", "cho_rej_min_std")


def _pissa_score(name: str, S_full: Tensor,
                 X: Tensor | dict[str, Tensor] | None,
                 Vh_full: Tensor, mode: str) -> Tensor:
    """Score every singular direction of W; caller takes top-r by score."""
    if mode == "s_only":
        return S_full
    if X is None:
        raise RuntimeError(f"selection_score={mode!r} needs calibration_activations[{name!r}]")

    if mode == "cho_rej_min_std":
        if not isinstance(X, dict) or "cho" not in X or "rej" not in X:
            raise RuntimeError(
                f"selection_score='cho_rej_min_std' needs "
                f"calibration_activations[{name!r}]={{'cho': X_cho, 'rej': X_rej}}")
        std_cho = _proj_std(X["cho"], Vh_full)                # (k,)
        std_rej = _proj_std(X["rej"], Vh_full)                # (k,)
        return torch.minimum(std_cho, std_rej)

    # Single-distribution modes.
    if not isinstance(X, Tensor):
        raise RuntimeError(
            f"selection_score={mode!r} needs calibration_activations[{name!r}]=Tensor")
    X_dev = X.to(Vh_full.device, dtype=torch.float32)
    proj = X_dev @ Vh_full.T                                  # (N, k)
    act = proj.abs().mean(dim=0)                              # (k,)
    if mode == "wanda":
        return S_full * act
    if mode == "act_only":
        return act
    raise ValueError(f"selection_score must be one of {_PISSA_SCORES}, got {mode!r}")


def _proj_std(X: Tensor, Vh_full: Tensor) -> Tensor:
    """std along N of X @ Vh_full.T, on Vh_full's device. X: (N, d_in)."""
    X_dev = X.to(Vh_full.device, dtype=torch.float32)
    proj = X_dev @ Vh_full.T                                  # (N, k)
    return proj.std(dim=0)                                    # (k,)


class ModulatedPiSSA:
    """Bidirectional SVD-space adapter with scalar coefficient `c`.

    Same surface as `ModulatedLoRA`: parameters(), set_coeff(), `c` property,
    `with adapter(model, c=...):` context manager, save/load/from_checkpoint.
    Per target Linear (W: d_out × d_in):
        Init     : SVD(W); top-r selected by `selection_score`; W mutated to
                   W_res = W - U_r·S_r·Vh_r; (U_r, S_r, Vh_r) stored as
                   buffers; Δs ∈ ℝ^r trainable, init N(4e-4, 4e-4).
        Forward  : y = layer(x) + U · ((S + c·Δs) ⊙ (x @ Vh.T)) @ U.T-shape...
                   (concretely: F.linear(F.linear(x, Vh) * (S+c·Δs), U))

    PiSSA quirk vs LoRA: `c=0` is NOT a hook short-circuit — every forward
    pays one rank-r matmul to reconstruct U·S·Vh·x (otherwise the layer
    would output W_res·x ≠ W·x). This is the cost of physically extracting
    the top-r out of W.
    """

    def __init__(self, model: nn.Module, r: int = 256,
                 targets: tuple[str, ...] = ("all-linear",),
                 layer_range: tuple[float, float] = (0.0, 1.0),
                 dtype: torch.dtype = torch.bfloat16,
                 calibration_activations: dict[str, Tensor | dict[str, Tensor]] | None = None,
                 selection_score: Literal["s_only", "wanda", "act_only",
                                          "cho_rej_min_std"] = "cho_rej_min_std",
                 _skip_init: bool = False):
        """`_skip_init=True` skeleton-mode for `from_checkpoint`; caller is
        responsible for filling buffers + delta_s and re-mutating layer.weight."""
        self.cfg = LoRAConfig(r=r, alpha=1.0, targets=targets,
                              layer_range=layer_range, dtype=dtype)
        self.selection_score = selection_score
        self._handles: list = []
        self._c: float = 0.0
        self._attached: bool = False

        device = next(model.parameters()).device
        targets_found = _find_targets(model, self.cfg)
        # Reject quantized layers — W mutation isn't reversible on bnb buffers.
        for name, layer in targets_found:
            if type(layer).__name__ in ("Linear4bit", "Linear8bitLt"):
                raise RuntimeError(
                    f"ModulatedPiSSA needs float nn.Linear; {name} is {type(layer).__name__}. "
                    "Run with quant=None or use ModulatedLoRA for quantized models."
                )

        self.U: dict[str, Tensor] = {}
        self.S: dict[str, Tensor] = {}
        self.Vh: dict[str, Tensor] = {}
        self.delta_s: dict[str, nn.Parameter] = {}
        self._target_layers: dict[str, nn.Linear] = {}
        self._chosen_idx: dict[str, Tensor] = {}    # for diagnostic logging

        for name, layer in targets_found:
            self._target_layers[name] = layer
            d_in, d_out = layer.in_features, layer.out_features
            # Per-target clamp: r > full-rank k for this layer (e.g. r=2304
            # against GQA k,v with min(d_in,d_out)=1024) just means "use full
            # spectrum here". Lets one cfg.r serve as "full per target" via
            # any large sentinel.
            r_used = min(r, min(d_in, d_out))
            if _skip_init:
                # Allocate empty skeletons; from_checkpoint fills them then
                # mutates layer.weight = W - U·S·Vh.
                self.U[name] = torch.empty(d_out, r_used, dtype=dtype, device=device)
                self.S[name] = torch.empty(r_used, dtype=dtype, device=device)
                self.Vh[name] = torch.empty(r_used, d_in, dtype=dtype, device=device)
                self.delta_s[name] = nn.Parameter(torch.empty(r_used, dtype=dtype, device=device))
                continue

            # Full SVD in float32 for numerical stability, then score + slice.
            with torch.no_grad():
                W = layer.weight.data.to(torch.float32)
                U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
                X = (calibration_activations or {}).get(name)
                scores = _pissa_score(name, S_full, X, Vh_full, selection_score)
                idx = scores.argsort(descending=True)[:r_used]
                idx = idx.sort().values                         # stable order
                Ur = U_full[:, idx].contiguous()
                Sr = S_full[idx].contiguous()
                Vhr = Vh_full[idx].contiguous()
                W_res = (W - (Ur * Sr) @ Vhr).to(layer.weight.dtype)
                layer.weight.data.copy_(W_res)
            self.U[name] = Ur.to(dtype).to(device).detach()
            self.S[name] = Sr.to(dtype).to(device).detach()
            self.Vh[name] = Vhr.to(dtype).to(device).detach()
            self._chosen_idx[name] = idx.cpu()
            # Δs init asymmetric (mean=std), 100× larger than the previous
            # 4e-4 timid init. Rationale: at init, the c=+C forward gives
            # (S + c·Δs); to give the optimizer real leverage on each
            # singular direction at c~1, |c·Δs| should be a meaningful
            # fraction of S (here mean(Δs)/mean(S) ≈ a few %, depending on
            # layer). The previous 4e-4 init meant c=2·Δs ≈ 8e-4 — well
            # below bf16 mantissa noise on top-r S values (S₀~tens) and the
            # adapter just sat in the SVD round-trip floor.
            # Asymmetric (mean=std>0) keeps a slight positive prior so c=+C
            # systematically amplifies and c=-C systematically attenuates
            # the chosen singular directions, preserving the +C/-C duality.
            ds = torch.empty(r_used, dtype=dtype, device=device).normal_(mean=4e-2, std=4e-2)
            self.delta_s[name] = nn.Parameter(ds)

        for p in model.parameters():
            p.requires_grad_(False)
        n_train = sum(p.numel() for p in self.parameters())
        logger.debug(f"ModulatedPiSSA: {len(targets_found)} targets, r={r}, "
                     f"selection={selection_score}, trainable={n_train:,}")
        if not _skip_init and self._chosen_idx:
            self._log_selection_diagnostic()

    def _log_selection_diagnostic(self) -> None:
        """For each target, summarize where in S the chosen indices fell.
        For pure top-S selection, mean(idx) = (r-1)/2. Activation-driven
        modes should push picks deeper into the tail, increasing mean(idx).
        Report `tail_depth = (mean(idx) - (r-1)/2) / (k - r)` (0 = top-S
        only; 1 = picks all live past rank r, in the tail).
        SHOULD: tail_depth > 0.05 for activation-driven modes on real
        prompts. ELSE selection collapsed to top-S = scoring isn't doing
        work (X capture broken, OR activations dominantly align with the
        top of S — the failure mode this fork was built to detect)."""
        if not self._chosen_idx:
            return
        ranks_mean = []
        tail_depths = []
        for name, idx in self._chosen_idx.items():
            layer = self._target_layers[name]
            k_full = min(layer.in_features, layer.out_features)
            r = self.cfg.r
            mean_idx = idx.float().mean().item()
            top_s_mean = (r - 1) / 2
            denom = max(1.0, k_full - r)
            tail_depths.append((mean_idx - top_s_mean) / denom)
            ranks_mean.append(mean_idx)
        m = sum(ranks_mean) / len(ranks_mean)
        td = sum(tail_depths) / len(tail_depths)
        logger.info(
            f"ModulatedPiSSA selection: mean(chosen_idx)={m:.1f} "
            f"tail_depth={td:.3f} across {len(ranks_mean)} targets "
            f"(r={self.cfg.r}, mode={self.selection_score}). "
            f"SHOULD be tail_depth > 0.05 for activation-driven modes; "
            f"≈0 means picks collapsed onto the top of S."
        )

    def parameters(self):
        for p in self.delta_s.values():
            yield p

    def _make_hook(self, name: str):
        U: Float[Tensor, "o r"] = self.U[name]
        Vh: Float[Tensor, "r i"] = self.Vh[name]
        S: Float[Tensor, "r"] = self.S[name]
        delta_s: Float[Tensor, "r"] = self.delta_s[name]

        def hook(layer: nn.Linear, args, y: Tensor) -> Tensor:
            (x,) = args
            xV = F.linear(x.to(Vh.dtype), Vh)               # (..., r) = x @ Vh.T
            scaled = xV * (S + self._c * delta_s)           # diag(S + c·Δs)
            delta = F.linear(scaled, U)                     # (..., d_out)
            return y + delta.to(y.dtype)

        return hook

    @contextmanager
    def __call__(self, model: nn.Module, c: float = 1.0):
        """`with adapter(model, c=...):` — hook adds U·(S+c·Δs)·Vh·x at every
        forward. c=0 reconstructs the original W (modulo SVD round-trip)."""
        if self._attached:
            raise RuntimeError("ModulatedPiSSA already attached; exit outer `with` first")
        self._c = float(c)
        for name, layer in self._target_layers.items():
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        self._attached = True
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            self._attached = False
            self._c = 0.0

    def set_coeff(self, c: float) -> None:
        self._c = float(c)

    @property
    def c(self) -> float:
        return self._c

    def restore_base_W(self) -> None:
        """layer.weight ← W_res + U·diag(S)·Vh. Required before any `baked()`
        use on a PiSSA-trained model (bake expects fresh W, not W_res)."""
        for name, layer in self._target_layers.items():
            with torch.no_grad():
                W_res = layer.weight.data.to(torch.float32)
                U32 = self.U[name].to(torch.float32)
                S32 = self.S[name].to(torch.float32)
                Vh32 = self.Vh[name].to(torch.float32)
                W = (W_res + (U32 * S32) @ Vh32).to(layer.weight.dtype)
                layer.weight.data.copy_(W)

    # ---- save / load -------------------------------------------------------

    def save(self, path: str, extra_meta: dict[str, str] | None = None) -> None:
        from safetensors.torch import save_file
        sd: dict[str, Tensor] = {}
        for k in self.U:
            kk = k.replace(".", "__")
            sd[f"U.{kk}"] = self.U[k].detach().cpu()
            sd[f"S.{kk}"] = self.S[k].detach().cpu()
            sd[f"Vh.{kk}"] = self.Vh[k].detach().cpu()
            sd[f"delta_s.{kk}"] = self.delta_s[k].detach().cpu()
        meta = {
            "kind": "pissa",
            "r": str(self.cfg.r),
            "targets": ",".join(self.cfg.targets),
            "layer_range": f"{self.cfg.layer_range[0]},{self.cfg.layer_range[1]}",
            "selection_score": self.selection_score,
        }
        if extra_meta:
            meta.update(extra_meta)
        save_file(sd, path, metadata=meta)

    def load(self, path: str) -> None:
        """Load buffers + delta_s AND mutate layer.weight to W - U·S·Vh.

        Assumes layer.weight currently holds the W that the adapter was
        trained against (i.e. base W for round 1; the W after round-(k-1)'s
        load for round k — caller must load in round order)."""
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
        ckpt_keys = {k[2:].replace("__", ".") for k in sd if k.startswith("U.")}
        init_keys = set(self.U.keys())
        if ckpt_keys != init_keys:
            raise RuntimeError(
                f"PiSSA target mismatch: checkpoint has {len(ckpt_keys)} targets, "
                f"init created {len(init_keys)}.")
        for k in self.U:
            kk = k.replace(".", "__")
            dev = self.U[k].device
            self.U[k].data.copy_(sd[f"U.{kk}"].to(dev, self.U[k].dtype))
            self.S[k].data.copy_(sd[f"S.{kk}"].to(dev, self.S[k].dtype))
            self.Vh[k].data.copy_(sd[f"Vh.{kk}"].to(dev, self.Vh[k].dtype))
            self.delta_s[k].data.copy_(sd[f"delta_s.{kk}"].to(dev, self.delta_s[k].dtype))
            # Mutate layer.weight: W -> W - U·S·Vh (= W_res that training saw).
            layer = self._target_layers[k]
            with torch.no_grad():
                W = layer.weight.data.to(torch.float32)
                U32 = self.U[k].to(torch.float32)
                S32 = self.S[k].to(torch.float32)
                Vh32 = self.Vh[k].to(torch.float32)
                W_res = (W - (U32 * S32) @ Vh32).to(layer.weight.dtype)
                layer.weight.data.copy_(W_res)

    @classmethod
    def from_checkpoint(cls, model: nn.Module, path: str) -> "ModulatedPiSSA":
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
        if meta.get("kind") != "pissa":
            raise RuntimeError(f"not a PiSSA checkpoint: kind={meta.get('kind')!r}")
        targets = tuple(meta["targets"].split(","))
        lr_meta = meta.get("layer_range", "0.0,1.0")
        lo, hi = (float(v) for v in lr_meta.split(","))
        adapter = cls(model, r=int(meta["r"]), targets=targets,
                      layer_range=(lo, hi),
                      dtype=next(model.parameters()).dtype,
                      selection_score=meta["selection_score"],
                      _skip_init=True)
        adapter.load(path)
        return adapter


# ---------------------------------------------------------------------------
# HistoryBake — kept adapters compose via a single gated forward hook.
# Storage: per-target concat A_cat = [A_1; A_2; …], B_cat = [s_1·B_1, …]
# so dW_combined = B_cat @ A_cat exactly, computed without materialising dW.
# Gate predicate set by training code: `lambda: lora._c != 0.0` makes the
# c=0 reference forward return pristine base.
# ---------------------------------------------------------------------------

class HistoryBake:
    def __init__(self, model: nn.Module, history: list[tuple["ModulatedLoRA", float]]):
        self._is_active = lambda: True       # inference default; train code overrides
        target_layers: dict[str, nn.Linear] = {}
        for lora, _ in history:
            for name, layer in lora._target_layers.items():
                target_layers.setdefault(name, layer)
        self._target_layers = target_layers

        r0 = history[0][0].cfg.r
        for lora, _ in history[1:]:
            assert lora.cfg.r == r0, f"kept-history rank mismatch: {lora.cfg.r} vs {r0}"
        target_dtype = history[0][0].cfg.dtype

        with torch.no_grad():
            A_cat: dict[str, Tensor] = {}
            B_cat: dict[str, Tensor] = {}
            for name, layer in target_layers.items():
                A_parts, B_parts = [], []
                for lora, c in history:
                    if name not in lora.A:
                        continue
                    s = c * lora.cfg.alpha / lora.cfg.r
                    A_parts.append(lora.A[name].to(target_dtype))
                    B_parts.append((s * lora.B[name]).to(target_dtype))
                A_cat[name] = torch.cat(A_parts, dim=0).to(layer.weight.device).detach()
                B_cat[name] = torch.cat(B_parts, dim=1).to(layer.weight.device).detach()
        self._A_cat = A_cat
        self._B_cat = B_cat

        self._handles = []
        for name, layer in target_layers.items():
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        Nr = len(history) * r0
        logger.info(f"HistoryBake: {len(history)} kept adapter(s), r_total={Nr}")

    def _make_hook(self, name: str):
        # Read A/B inside the hook body — closure captures `self`, not the
        # tensors. Safe against future push/pop replacement of A_cat/B_cat.
        def hook(layer: nn.Linear, args, y: Tensor) -> Tensor:
            if not self._is_active():
                return y
            (x,) = args
            A = self._A_cat[name]                     # (k_total, d_in)
            B = self._B_cat[name]                     # (d_out, k_total), scale pre-folded
            xA = x.to(A.dtype)
            h = F.linear(xA, A)                       # x @ A.T via cuBLAS
            delta = F.linear(h, B)                    # h @ B.T via cuBLAS
            return y + delta.to(y.dtype)

        return hook

    def set_gate(self, is_active) -> None:
        """is_active() -> bool. Training: `set_gate(lambda: lora._c != 0.0)`."""
        self._is_active = is_active

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# PiSSAHistoryBake — composes kept ModulatedPiSSA rounds.
#
# Each kept round k has already mutated layer.weight by subtracting
# U_k·S_k·Vh_k. Without reconstruction the forward returns W_res_K·x ≠ W·x,
# i.e. nonsense — so this hook is ALWAYS active (no gate). It sums each
# round's contribution at its kept c_k:
#     Σ_k  U_k · diag(S_k + c_k · Δs_k) · Vh_k · x
#
# `set_gate` is a no-op kept for ModulatedLoRA-interface parity. The KL
# anchor semantic shifts from "pristine base" to "prior-baked state" —
# c=0 on the *current* round gives W_res_K + Σ_prior reconstructions,
# i.e. the state after all kept rounds at their kept coefficients. See
# CLAUDE.md note on KL.
# ---------------------------------------------------------------------------

class PiSSAHistoryBake:
    def __init__(self, model: nn.Module,
                 history: list[tuple["ModulatedPiSSA", float]]):
        if not history:
            raise ValueError("PiSSAHistoryBake needs ≥1 kept round")
        target_layers: dict[str, nn.Linear] = {}
        for adapter, _ in history:
            for name, layer in adapter._target_layers.items():
                target_layers.setdefault(name, layer)
        self._target_layers = target_layers
        target_dtype = history[0][0].cfg.dtype

        # Per layer: stack per-round (U_k, Vh_k) along the rank axis exactly
        # like HistoryBake stacks (A_k, B_k). The diag(S_k + c_k·Δs_k) factor
        # is folded into a single per-round scale vector — concat into one
        # (Σr,) vector aligned with the stacked rank axis.
        U_cat: dict[str, Tensor] = {}
        Vh_cat: dict[str, Tensor] = {}
        scale_cat: dict[str, Tensor] = {}
        with torch.no_grad():
            for name, layer in target_layers.items():
                U_parts, Vh_parts, scale_parts = [], [], []
                for adapter, c in history:
                    if name not in adapter.U:
                        continue
                    U_parts.append(adapter.U[name].to(target_dtype))
                    Vh_parts.append(adapter.Vh[name].to(target_dtype))
                    s = adapter.S[name].to(target_dtype) + float(c) * adapter.delta_s[name].detach().to(target_dtype)
                    scale_parts.append(s)
                if not U_parts:
                    continue
                U_cat[name] = torch.cat(U_parts, dim=1).to(layer.weight.device).detach()
                Vh_cat[name] = torch.cat(Vh_parts, dim=0).to(layer.weight.device).detach()
                scale_cat[name] = torch.cat(scale_parts, dim=0).to(layer.weight.device).detach()
        self._U_cat = U_cat
        self._Vh_cat = Vh_cat
        self._scale_cat = scale_cat

        self._handles = []
        for name, layer in target_layers.items():
            if name in U_cat:
                self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        r0 = history[0][0].cfg.r
        Nr = len(history) * r0
        logger.info(f"PiSSAHistoryBake: {len(history)} kept adapter(s), r_total={Nr}")

    def _make_hook(self, name: str):
        def hook(layer: nn.Linear, args, y: Tensor) -> Tensor:
            (x,) = args
            U = self._U_cat[name]                          # (d_out, Σr)
            Vh = self._Vh_cat[name]                        # (Σr, d_in)
            s = self._scale_cat[name]                      # (Σr,)
            xV = F.linear(x.to(Vh.dtype), Vh)              # (..., Σr)
            scaled = xV * s
            delta = F.linear(scaled, U)                    # (..., d_out)
            return y + delta.to(y.dtype)
        return hook

    def set_gate(self, _is_active) -> None:
        """No-op for PiSSA: kept-round reconstructions must always be active
        (otherwise layer.weight = W_res ≠ W and the forward returns garbage).
        Kept for ModulatedLoRA-interface parity with the train loop."""
        pass

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
