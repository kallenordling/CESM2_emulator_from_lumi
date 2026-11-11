"""Alternative denoising objectives for the loss-function arms.

Every arm trains the same v-prediction network on the same data; only the way
the per-pixel residual ``e = model_output - target`` is turned into a scalar
changes. Three independent knobs, all off by default, so the default config is
exactly the cos-latitude-weighted MSE every released run used:

``loss_type``
    ``mse``           e^2                                   (default)
    ``pseudo_huber``  2c (sqrt(e^2 + c^2) - c)              (``huber_c``)
                      = e^2 for |e| << c and 2c|e| beyond, so gradients match
                      the MSE arm on typical residuals and only the tails
                      (PRECT extremes, rare outlier members) are down-weighted.
                      Unlike the other two knobs this CHANGES the optimum: it
                      moves from the conditional mean towards the median.

``snr_weighting``
    ``none``          every noise level weighted as plain v-MSE (default)
    ``min_snr``       Min-SNR-gamma (Hang et al. 2023): per-sample weight
                      min(SNR, gamma) / (SNR + 1) on the v-loss, i.e. x0-loss
                      weight min(SNR, gamma). Relative to plain v-MSE (x0 weight
                      SNR + 1) this down-weights BOTH ends: nearly-clean steps
                      (SNR > gamma, weight gamma/(SNR+1)) and the very noisy
                      steps (weight ~SNR, e.g. 5e-5 at log-SNR -10, as an
                      epsilon loss would). Capacity moves to the mid-SNR band
                      (log-SNR ~ -2..+2). Note that the noisy steps are where
                      the forced large-scale mean E[x0 | cond] is learned, so
                      this arm tests whether that band is over- or under-served.
    ``elbo``          maximum likelihood: the continuous-time ELBO (Kingma et
                      al. 2021), i.e. the cross-entropy between CESM2 and the
                      emulator up to a constant. As an epsilon loss it is
                      uniform in log-SNR; as a v loss its density is
                      sigmoid(log-SNR). Rather than weighting uniform-t samples
                      by -dlambda/dt * sigmoid(lambda), which spans ~5 orders
                      of magnitude and makes the gradient noisy, log-SNR is
                      SAMPLED from that density (importance sampling, see
                      sample_elbo_log_snr) and every sample is weighted 1.
                      Only ~7.5% of samples then land at log-SNR < 0: the
                      opposite emphasis to min_snr. Rewards a calibrated
                      ensemble spread rather than sharp large-scale means.

``multiscale_weight`` (+ ``multiscale_pools``)
    Adds, for each pooling size k, the MSE of the area-weighted k x k block
    mean of the residual. A spatially white error averages away in the pooled
    term; a coherent regional error does not, so with weight w and n scales a
    coherent error counts up to (1 + w n) times as much as a white one. This
    targets the regional, large-scale response (e.g. the differential aerosol
    signal) without the global-mean x0-proxy penalty that failed before:
    every term is a positive quadratic form in the residual, so the optimum is
    still E[v | x_t, cond]; only how the capacity is spent changes.

Latitude weighting (cos lat, clamped at 0.2) and the per-target-channel weights
apply to every term exactly as in ``calc_mse_loss``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

LOSS_TYPES = ("mse", "pseudo_huber")
SNR_WEIGHTINGS = ("none", "min_snr", "elbo")


@dataclass(frozen=True)
class LossConfig:
    loss_type: str = "mse"
    huber_c: float = 0.05
    snr_weighting: str = "none"
    min_snr_gamma: float = 5.0
    multiscale_weight: float = 0.0
    multiscale_pools: tuple = field(default=(4, 16, 48))

    @classmethod
    def from_trainer(cls, obj) -> "LossConfig":
        """Read the knobs off a trainer (hyperparameters are set as attributes)."""
        d = cls()
        pools = getattr(obj, "multiscale_pools", d.multiscale_pools)
        cfg = cls(
            loss_type=str(getattr(obj, "loss_type", d.loss_type)),
            huber_c=float(getattr(obj, "huber_c", d.huber_c)),
            snr_weighting=str(getattr(obj, "snr_weighting", d.snr_weighting)),
            min_snr_gamma=float(getattr(obj, "min_snr_gamma", d.min_snr_gamma)),
            multiscale_weight=float(getattr(obj, "multiscale_weight", d.multiscale_weight)),
            multiscale_pools=tuple(int(k) for k in pools),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type={self.loss_type!r}; expected one of {LOSS_TYPES}")
        if self.snr_weighting not in SNR_WEIGHTINGS:
            # YAML reads a bare `off`/`no` as a BOOL, which str() turns into
            # 'False' -- catch it here rather than silently training plain MSE.
            raise ValueError(
                f"snr_weighting={self.snr_weighting!r}; expected one of "
                f"{SNR_WEIGHTINGS} (quote it in YAML)")
        if self.huber_c <= 0:
            raise ValueError(f"huber_c must be > 0, got {self.huber_c}")
        if self.min_snr_gamma <= 0:
            raise ValueError(f"min_snr_gamma must be > 0, got {self.min_snr_gamma}")
        if self.multiscale_weight < 0:
            raise ValueError(f"multiscale_weight must be >= 0, got {self.multiscale_weight}")
        if self.multiscale_weight > 0 and not self.multiscale_pools:
            raise ValueError("multiscale_weight > 0 needs at least one pool size")
        if any(k < 2 for k in self.multiscale_pools):
            raise ValueError(f"multiscale_pools must be >= 2, got {self.multiscale_pools}")

    @property
    def is_plain_mse(self) -> bool:  # elbo changes the SAMPLER, so it is not plain
        return (self.loss_type == "mse" and self.snr_weighting == "none"
                and self.multiscale_weight == 0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["multiscale_pools"] = list(self.multiscale_pools)
        return d

    def describe(self) -> str:
        parts = [self.loss_type + (f"(c={self.huber_c:g})" if self.loss_type == "pseudo_huber" else "")]
        if self.snr_weighting == "min_snr":
            parts.append(f"min_snr(gamma={self.min_snr_gamma:g})")
        if self.snr_weighting == "elbo":
            parts.append("elbo(log-SNR ~ sigmoid)")
        if self.multiscale_weight > 0:
            parts.append(f"multiscale(w={self.multiscale_weight:g}, pools={list(self.multiscale_pools)})")
        return " + ".join(parts)


def sample_elbo_log_snr(n: int, log_snr_min: float, log_snr_max: float,
                        *, device=None, generator=None) -> torch.Tensor:
    """Draw log-SNR with density proportional to sigmoid(lambda) on [min, max].

    That is the v-loss ELBO weight after the change of variable t -> lambda:
    -dlambda/dt * sigmoid(lambda) dt = sigmoid(lambda) dlambda, independent of
    the noise schedule. The CDF is softplus, so invert it: lambda =
    log(expm1(y)) with y uniform between softplus(min) and softplus(max).
    """
    lo = F.softplus(torch.tensor(float(log_snr_min), dtype=torch.float64))
    hi = F.softplus(torch.tensor(float(log_snr_max), dtype=torch.float64))
    u = torch.rand(n, dtype=torch.float64, device=device, generator=generator)
    y = lo.to(u.device) + u * (hi - lo).to(u.device)
    return torch.log(torch.expm1(y)).float()


def snr_weights(log_snr: torch.Tensor, cfg: LossConfig, prediction_type: str) -> torch.Tensor | None:
    """Per-sample weight (B,) from the noise level, or None for uniform."""
    if cfg.snr_weighting in ("none", "elbo"):   # elbo: importance-sampled, weight 1
        if cfg.snr_weighting == "elbo" and prediction_type != "v_prediction":
            raise ValueError("elbo sampling is derived for v_prediction")
        return None
    snr = torch.exp(log_snr.float())
    clipped = torch.clamp(snr, max=cfg.min_snr_gamma)
    if prediction_type == "v_prediction":
        return clipped / (snr + 1.0)
    if prediction_type == "epsilon":
        return clipped / snr
    raise ValueError(f"min_snr weighting not defined for prediction_type={prediction_type!r}")


def _pixel_loss(e: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    if cfg.loss_type == "mse":
        return e * e
    c = cfg.huber_c
    return 2.0 * c * (torch.sqrt(e * e + c * c) - c)


def _pool_area_mean(x: torch.Tensor, area: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-weighted k x k block means of x (..., H, W), and the block areas."""
    H, W = x.shape[-2:]
    if H % k or W % k:
        raise ValueError(f"multiscale pool {k} does not divide the grid {H}x{W}")
    lead = x.shape[:-2]
    xa = (x * area).reshape(-1, 1, H, W)
    num = F.avg_pool2d(xa, k)
    den = F.avg_pool2d(area.reshape(1, 1, H, W), k)
    return (num / den).reshape(*lead, H // k, W // k), den.reshape(H // k, W // k)


def denoising_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    log_snr: torch.Tensor,
    lat_weight: torch.Tensor,
    cfg: LossConfig,
    *,
    prediction_type: str = "v_prediction",
    channel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """The training objective for one batch.

    model_output, target : (B, C, T, H, W)
    log_snr              : (B,) noise level of each sample
    lat_weight           : (H,) cos(lat).clamp(0.2), NOT mean-normalised --
                           the same weight calc_mse_loss uses, so the plain-MSE
                           value logged by every arm is directly comparable.
    Returns (loss, components) where components holds detached scalars.
    """
    e = model_output.float() - target.float()
    B = e.shape[0]
    lw = lat_weight.float().view(-1, 1)                       # (H, 1)
    cw = None if channel_weights is None else channel_weights.float()
    tw = snr_weights(log_snr, cfg, prediction_type)
    tw = None if tw is None else tw.view(B, *([1] * (e.dim() - 1)))

    def _reduce(per_pix: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        if cw is not None:
            per_pix = per_pix * cw
        if tw is not None:
            per_pix = per_pix * tw
        return (per_pix * area).mean()

    loss = _reduce(_pixel_loss(e, cfg), lw)
    comps = {"loss/pixel": loss.detach()}

    if cfg.multiscale_weight > 0:
        area = lw.expand(e.shape[-2], e.shape[-1])
        for k in cfg.multiscale_pools:
            pooled, parea = _pool_area_mean(e, area, k)
            term = _reduce(pooled * pooled, parea)
            comps[f"loss/pool{k}"] = term.detach()
            loss = loss + cfg.multiscale_weight * term
    return loss, comps
