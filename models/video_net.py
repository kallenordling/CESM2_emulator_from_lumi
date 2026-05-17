import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops_exts import rearrange_many
from einops import rearrange

from torch.utils import checkpoint as torch_checkpoint

from models.rotary_embedding import RotaryEmbedding


class LonCircularConv3d(nn.Module):
    """Conv3d with circular padding along the longitude (W) dimension.

    Longitude is periodic (0° == 360°), so zero-padding at the left/right
    edges introduces a spurious discontinuity.  This wrapper manually applies
    circular padding along W before the convolution, while keeping the normal
    zero-padding along H (latitude, which is NOT periodic).

    Weight shapes are identical to the equivalent nn.Conv3d, so existing
    checkpoints load without modification.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__()
        # Normalise to 3-tuples
        def _t(v): return (v, v, v) if isinstance(v, int) else tuple(v)
        kernel_size = _t(kernel_size)
        stride      = _t(stride)
        padding     = _t(padding)

        self.lon_pad = padding[2]   # circular padding amount for W
        # Hand off T and H padding to Conv3d; W padding is done manually
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=(padding[0], padding[1], 0),
            **kwargs,
        )

    def forward(self, x):
        if self.lon_pad > 0:
            x = F.pad(x, (self.lon_pad, self.lon_pad, 0, 0, 0, 0), mode="circular")
        return self.conv(x)


def checkpoint(fn, *args, enabled=False):
    if enabled:
        # torch.compiler.disable prevents dynamo from tracing *into* the
        # checkpointed block.  Without this, torch.compile + gradient
        # checkpointing on ROCm (MI250X) produces a signal-11 segfault because
        # dynamo tries to inline through the remat boundary and trips over
        # autograd profiler hooks that checkpoint registers internally.
        # The checkpointed blocks still run correctly (eagerly inside the
        # compiled graph); only the recomputation during backward is affected,
        # which is the desired behaviour anyway.
        @torch.compiler.disable
        def _run_checkpointed():
            return torch_checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return _run_checkpointed()
    else:
        return fn(*args)


# Convert value to an n-element tuple of value
def cast_to_tuple(val, n):
    if isinstance(val, tuple):
        return val
    elif isinstance(val, list):
        return tuple(val)
    elif isinstance(val, int) or isinstance(val, bool):
        return [val] * n
    else:
        return tuple(val)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def Downsample(dim):
    return LonCircularConv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Compute the frequency table in fp32 (the exp covers a wide dynamic
        # range that bf16/fp16 quantise poorly), then cast the result back to
        # the caller's dtype so the rest of the network can run in bf16/fp16.
        device = x.device
        out_dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(out_dtype)


class Pseudo3DConv(nn.Module):
    def __init__(
            self, dim, *, kernel_size, dim_out=None, temporal_kernel_size=None, **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(
            dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.temporal_conv = nn.Conv1d(
            dim_out,
            dim_out,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        )

        nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, convolve_across_time=True):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        convolve_across_time &= is_video

        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b)

        if not convolve_across_time:
            return x

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


class Cond_2D_CNN(nn.Module):
    """A CNN that processes a conditioning map to turn it into an embedding"""

    def __init__(self, time_emb_dim, image_size, kernel_size=3, out_channels=8):
        # Initialize as module
        super().__init__()

        # Create the convolution,
        self.in_conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel_size, padding=1
        )

        self.relu = nn.SiLU()

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=1
        )

        # Padding amount will be half of the kernel size
        self.pad = kernel_size // 2

        # Calculate the dimensionality of the in features to the linear network
        self.lin_in_features = image_size[0] * image_size[1] * out_channels

        # Create linear projection to time embedding dimension
        self.linear_proj = nn.Linear(self.lin_in_features, time_emb_dim)

        self.res_conv = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)

        # Send x through the convolutional layer and linear projection
        h = self.relu(self.norm1(self.in_conv(x)))

        h = self.relu(self.norm2(self.conv2(h)))

        h = h + self.res_conv(x)

        h = rearrange(h, "b c h w -> b (h w c)")
        out = self.relu(self.linear_proj(h))

        return out


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = LonCircularConv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
            self, dim, dim_out, *, time_emb_dim=None, groups=8, use_checkpoint=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, spatial_film=None):
        if self.training:
            return checkpoint(self._forward, x, time_emb, spatial_film, enabled=self.use_checkpoint)
        else:
            return self._forward(x, time_emb, spatial_film)

    def _forward(self, x, time_emb=None, spatial_film=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        # spatial_film: [B, dim_out*2, T, H, W] — split into scale and shift once,
        # reuse for both sub-blocks so every layer has a direct gradient path.
        sp_scale, sp_shift = (None, None)
        if exists(spatial_film):
            sp_scale, sp_shift = spatial_film.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        if sp_scale is not None:
            h = h * (sp_scale + 1) + sp_shift

        h = self.block2(h)
        if sp_scale is not None:
            h = h * (sp_scale + 1) + sp_shift

        return h + self.res_conv(x)


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
                max_exact
                + (
                        torch.log(n.float() / max_exact)
                        / math.log(max_distance / max_exact)
                        * (num_buckets - max_exact)
                ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x, enabled=self.use_checkpoint)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs
        )
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, heads=4, dim_head=32, rotary_emb=None, use_checkpoint=False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        if self.training:
            return checkpoint(
                self._forward,
                x,
                pos_bias,
                focus_present_mask,
                enabled=self.use_checkpoint,
            )
        else:
            return self._forward(x, pos_bias, focus_present_mask)

    def _forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None,
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = torch.einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = torch.einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


class TemporalCNN(nn.Module):
    def __init__(self, dim, kernel_size=3, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.temporal_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2
        )

        nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, **kwargs):
        if self.training:
            return checkpoint(self._forward, x, enabled=self.use_checkpoint)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *_, h, w = x.shape

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)
        return x


class PseudoConv3D(nn.Module):
    def __init__(
            self, dim, *, kernel_size, dim_out=None, temporal_kernel_size=None, **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(
            dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.temporal_conv = nn.Conv1d(
            dim_out,
            dim_out,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        )

        nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, convolve_across_time=True):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        convolve_across_time &= is_video

        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b)

        if not convolve_across_time:
            return x

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


class SpatialCondEncoder(nn.Module):
    """
    Multi-scale conditioning encoder that preserves spatial structure.

    Takes [B, cond_channels, T, H, W] and produces a list of feature maps,
    one per UNet resolution level.  Unlike global average pooling, every spatial
    cell in the emission map retains a distinct representation that can modulate
    the corresponding cell in the UNet via per-pixel FiLM.

    Level i has shape [B, dims[i+1], T, H/2^i, W/2^i] (last level keeps size).
    Cross-level residual projections (zero-init) accumulate multi-scale context
    in deeper features and shorten the gradient path from deep UNet layers back
    to the conditioning encoder.
    """

    def __init__(self, cond_channels: int, dims: list, resnet_groups: int = 8):
        super().__init__()
        level_dims        = dims[1:]
        n_levels          = len(level_dims)
        self.levels       = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        in_ch = cond_channels
        for i, out_ch in enumerate(level_dims):
            g = min(resnet_groups, out_ch)
            self.levels.append(nn.Sequential(
                LonCircularConv3d(in_ch,   out_ch, (1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(g, out_ch),
                nn.SiLU(),
                LonCircularConv3d(out_ch, out_ch, (1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(g, out_ch),
                nn.SiLU(),
            ))
            # Match UNet Downsample; last level is identity (is_last)
            if i < n_levels - 1:
                self.downsamplers.append(
                    LonCircularConv3d(out_ch, out_ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
                )
            else:
                self.downsamplers.append(nn.Identity())

            # Zero-init cross-level residual: starts as no-op, model learns to use it
            if i == 0:
                self.residual_projs.append(None)
            else:
                prev_ch = level_dims[i - 1]
                res_proj = nn.Conv3d(prev_ch, out_ch, 1)
                nn.init.zeros_(res_proj.weight)
                nn.init.zeros_(res_proj.bias)
                self.residual_projs.append(res_proj)

            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> list:
        """Returns list of feature maps BEFORE downsampling, one per level."""
        features     = []
        prev_feat_ds = None
        for level, ds, res_proj in zip(self.levels, self.downsamplers, self.residual_projs):
            feat = level(x)
            if res_proj is not None and prev_feat_ds is not None:
                feat = feat + res_proj(prev_feat_ds)
            features.append(feat)
            prev_feat_ds = ds(feat)
            x = prev_feat_ds
        return features


class UNetModel3D(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            # n_vars,
            model_dim, in_channels, out_channels,
            dim_mults=(1, 2, 4, 8),
            attn_heads=8,
            attn_dim_head=32,
            use_sparse_linear_attn=True,
            use_mid_attn=False,
            init_kernel_size=7,
            resnet_groups=8,
            use_checkpoint=False,
            use_temp_attn=True,
            day_cond=False,
            year_cond=False,
            cond_map=True,
            cond_channels=0,
    ):
        super().__init__()

        self.use_temp_attn = use_temp_attn
        self.year_cond = year_cond
        self.day_cond = day_cond
        self.cond_channels = cond_channels

        # Input and output size to the model will be how many variables we are predicting
        # in_channels = n_vars
        # out_channels = n_vars

        # Add an input channel for the conditioning map
        # if cond_map:
        #    in_channels += n_vars

        # Initial convolution and attention to process input
        init_padding = init_kernel_size // 2
        self.input_conv = LonCircularConv3d(
            in_channels,
            model_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        # If we are using temporal attn over convolution
        if use_temp_attn:
            # Define positional encodings and a temporal attention constructor
            self.time_rel_pos_bias = RelativePositionBias(
                heads=attn_heads, max_distance=32
            )

            # Create temporal attention operation only just frames
            def temporal_op(dim):
                return EinopsToAndFrom(
                    "b c f h w",
                    "b (h w) f c",
                    Attention(
                        dim,
                        heads=attn_heads,
                        dim_head=attn_dim_head,
                        rotary_emb=rotary_emb,
                        use_checkpoint=use_checkpoint,
                    ),
                )

        else:
            # Otherwise use temporal convolutions only
            def temporal_op(dim):
                return TemporalCNN(dim, kernel_size=3, use_checkpoint=use_checkpoint)

        # Define positional encodings and a temporal attention constructor
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32
        )

        def temporal_attn(dim):
            return EinopsToAndFrom(
                "b c f h w",
                "b (h w) f c",
                Attention(
                    dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb
                ),
            )

        # Initial input temporal operation
        self.input_temp_op = Residual(PreNorm(model_dim, temporal_op(model_dim)))

        # Construct a list of tuples describing the in and out channels of each layer in model
        dims = [model_dim, *map(lambda m: int(model_dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Define time embedding dimension and create MLP to process timesteps
        time_dim = model_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Spatial conditioning encoder ──────────────────────────────────────
        # Replaces the old global-avg-pool scalar encoder.
        # Produces a feature pyramid [B, dims[i+1], T, H_i, W_i] per level.
        # Per-pixel FiLM is injected into both ResnetBlocks at every UNet level
        # via zero-initialised 1x1x1 projection convs so the model starts as the
        # unconditioned baseline and learns conditioning gradually.
        if cond_channels > 0:
            self.spatial_cond_encoder = SpatialCondEncoder(
                cond_channels=cond_channels,
                dims=dims,
                resnet_groups=resnet_groups,
            )

            # Down projections (block1 and block2 at each level)
            self.cond_down_projs = nn.ModuleList([
                nn.Conv3d(dim_out, dim_out * 2, 1)
                for (_, dim_out) in in_out
            ])
            self.cond_down_block1_projs = nn.ModuleList([
                nn.Conv3d(dim_out, dim_out * 2, 1)
                for (_, dim_out) in in_out
            ])

            # Up projections: spatial feat has dim_out channels, FiLM needs dim_in*2
            self.cond_up_projs = nn.ModuleList([
                nn.Conv3d(dim_out, dim_in * 2, 1)
                for (dim_in, dim_out) in reversed(in_out)
            ])
            self.cond_up_block1_projs = nn.ModuleList([
                nn.Conv3d(dim_out, dim_in * 2, 1)
                for (dim_in, dim_out) in reversed(in_out)
            ])

            # Bottleneck projections
            mid_dim_for_proj = dims[-1]
            self.cond_mid_proj        = nn.Conv3d(mid_dim_for_proj, mid_dim_for_proj * 2, 1)
            self.cond_mid_block1_proj = nn.Conv3d(mid_dim_for_proj, mid_dim_for_proj * 2, 1)

            # ── Direct input injection (always-on gradient path) ──────────────
            # Projects cond_map into the same space as the noisy input x so it
            # can be added BEFORE input_conv.  Spatial kernel 5×5 (time kernel 1)
            # forces the per-pixel injection to spatially average, suppressing
            # the leak of inventory line features (shipping lanes, flight paths)
            # straight into the prediction.  Padding keeps shape unchanged.
            self.cond_input_proj = nn.Conv3d(
                cond_channels, in_channels,
                kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=True,
            )
            nn.init.kaiming_normal_(self.cond_input_proj.weight, a=0.01)
            nn.init.zeros_(self.cond_input_proj.bias)

            # Zero-init all FiLM projections: identity at start, model learns gradually
            for proj in [
                *self.cond_down_projs, *self.cond_down_block1_projs,
                *self.cond_up_projs,   *self.cond_up_block1_projs,
                self.cond_mid_proj,    self.cond_mid_block1_proj,
            ]:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

            # Keep old names as None so external code that checks them doesn't crash
            self.cond_encoder = None
            self.cond_scale   = None
            self.cond_shift   = None
        else:
            self.spatial_cond_encoder   = None
            self.cond_input_proj        = None   # no direct injection without cond
            self.cond_down_projs        = None
            self.cond_down_block1_projs = None
            self.cond_up_projs          = None
            self.cond_up_block1_projs   = None
            self.cond_mid_proj          = None
            self.cond_mid_block1_proj   = None
            self.cond_encoder           = None
            self.cond_scale             = None
            self.cond_shift             = None
        # Create embeddings for day and year if applicable
        if day_cond:
            self.class_emb = nn.Embedding(366, time_dim)
        if year_cond:
            self.year_emb = nn.Embedding(252, time_dim)

        # Create lists to hold up and down blocks of U-Net
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # Partially Initialize resnet-blocks
        block_klass = partial(
            ResnetBlock, groups=resnet_groups, use_checkpoint=use_checkpoint
        )
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # Constructing down blocks of U-Net
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            has_attn = ind >= (num_resolutions - 3)

            # Downblock: 2 Residual Blocks, 1 spatial linear attention, 1 temporal attention, 1 downsample (at all but last levels)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass_cond(dim_in, dim_out),
                        block_klass_cond(dim_out, dim_out),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    SpatialLinearAttention(
                                        dim_out,
                                        heads=attn_heads,
                                        use_checkpoint=use_checkpoint,
                                    ),
                                )
                            )
                            if use_sparse_linear_attn or has_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_out, temporal_op(dim_out) if not has_attn else temporal_attn(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Construct bottleneck layers
        mid_dim = dims[-1]

        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        # Only do spatial attn on middle layer if we are using spatial attn
        if use_mid_attn:
            spatial_attn = EinopsToAndFrom(
                "b c f h w",
                "b f (h w) c",
                Attention(mid_dim, heads=attn_heads, use_checkpoint=use_checkpoint),
            )
            self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))

        else:
            self.mid_spatial_attn = nn.Identity()

        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # Construct Up Blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            has_attn = ind in [0, 1, 2]

            # Up Block: 2 Residual blocks, 1 spatial attention, 1 temporal attention, 1 upsampling layer
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass_cond(
                            dim_out * 2, dim_in
                        ),  # dim_out * 2 to account for incoming residual connection
                        block_klass_cond(dim_in, dim_in),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    SpatialLinearAttention(
                                        dim_in,
                                        heads=attn_heads,
                                        use_checkpoint=use_checkpoint,
                                    ),
                                )
                            )
                            if use_sparse_linear_attn or has_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_in, temporal_op(dim_in) if not has_attn else temporal_attn(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Output convolution to bring hidden back to original shape
        self.out_conv = nn.Sequential(
            block_klass(model_dim * 2, model_dim), nn.Conv3d(model_dim, out_channels, 1)
        )

    def forward(
            self,
            x,
            timesteps,
            days=None,
            years=None,
            cond_map=None,
            lowres_cond=None,
            focus_present_mask=None,
            prob_focus_present=0,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 1. time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # If we are using attn for temporal representation

        if exists(self.time_rel_pos_bias):
            # Create keyword arguments for attention operations
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
            focus_present_mask = default(
                focus_present_mask,
                lambda: prob_mask_like(
                    (x.shape[0],), prob_focus_present, device=x.device
                ),
            )

        # Otherwise set them to be none
        else:
            time_rel_pos_bias = None
            focus_present_mask = None


        # ── Spatial conditioning encoder ──────────────────────────────────────
        # Produces a list of feature maps at each UNet resolution.
        # cond_spatial_feats[i] has shape [B, dims[i+1], T, H_i, W_i]
        #
        # CFG dropout is handled entirely by the trainer BEFORE calling forward().
        # The trainer sets dropped rows to -1.0 (the correct null/pre-industrial
        # value under the normalisation scheme: zero emissions -> -1).
        # We must NOT apply a second dropout here: doing so would compound the
        # drop rate and, critically, would zero rows to 0.0 instead of -1.0,
        # teaching the model a wrong null conditioning value.
        NULL_COND_VALUE = -1.0
        cond_spatial_feats = None
        if self.spatial_cond_encoder is not None:
            if exists(cond_map):
                cond_map_input = cond_map
            else:
                # Null conditioning: CFG inference called with cond_map=None.
                # Fill with -1.0 (pre-industrial baseline), NOT 0.0.
                cond_map_input = torch.full(
                    (x.shape[0], self.cond_channels,
                     x.shape[2], x.shape[3], x.shape[4]),
                    NULL_COND_VALUE,
                    device=x.device, dtype=x.dtype,
                )
            cond_spatial_feats = self.spatial_cond_encoder(cond_map_input)

        if exists(lowres_cond):
            x = torch.cat([x, lowres_cond], dim=1)

        # ── Direct input injection — always-on conditioning path ──────────────
        # Projects cond_map into input space and adds it to x before input_conv.
        # This gives an immediate, zero-lag gradient path from loss → cond_map,
        # active from step 1 regardless of the FiLM projection warm-up.
        if self.cond_input_proj is not None and self.spatial_cond_encoder is not None:
            x = x + self.cond_input_proj(cond_map_input)

        # Send x through first convolution and temporal operation
        x = self.input_conv(x)
        x = self.input_temp_op(x, pos_bias=time_rel_pos_bias)

        # Create a residual connection to add at the end of the model's forward process
        r = x.clone()

        # Get timestep embeddings (carries diffusion noise level only; no cond injection)
        t = self.time_mlp(timesteps)

        if self.day_cond:
            t += self.class_emb(days)
        if self.year_cond:
            t += self.year_emb(years)

        # Store skip connections
        h = []

        # ── Down blocks ───────────────────────────────────────────────────────
        for level_idx, (block1, block2, spatial_attn, temporal_attn, downsample) in enumerate(self.downs):
            sp_film_b1 = None
            sp_film_b2 = None
            if cond_spatial_feats is not None:
                feat = cond_spatial_feats[level_idx]
                if self.cond_down_block1_projs is not None:
                    sp_film_b1 = self.cond_down_block1_projs[level_idx](feat)
                if self.cond_down_projs is not None:
                    sp_film_b2 = self.cond_down_projs[level_idx](feat)

            x = block1(x, t, spatial_film=sp_film_b1)
            x = block2(x, t, spatial_film=sp_film_b2)
            x = spatial_attn(x)
            x = temporal_attn(
                x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
            )
            h.append(x)
            x = downsample(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        sp_film_mid_b1 = None
        sp_film_mid_b2 = None
        if cond_spatial_feats is not None:
            feat = cond_spatial_feats[-1]
            if self.cond_mid_block1_proj is not None:
                sp_film_mid_b1 = self.cond_mid_block1_proj(feat)
            if self.cond_mid_proj is not None:
                sp_film_mid_b2 = self.cond_mid_proj(feat)

        x = self.mid_block1(x, t, spatial_film=sp_film_mid_b1)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
        )
        x = self.mid_block2(x, t, spatial_film=sp_film_mid_b2)

        # ── Up blocks ─────────────────────────────────────────────────────────
        num_down_levels = len(self.downs)
        for up_idx, (block1, block2, spatial_attn, temporal_attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)

            sp_film_b1 = None
            sp_film_b2 = None
            if cond_spatial_feats is not None:
                # Mirror down path: up_idx=0 -> deepest down level
                down_level_idx = num_down_levels - 1 - up_idx
                feat = cond_spatial_feats[down_level_idx]
                if self.cond_up_block1_projs is not None:
                    sp_film_b1 = self.cond_up_block1_projs[up_idx](feat)
                if self.cond_up_projs is not None:
                    sp_film_b2 = self.cond_up_projs[up_idx](feat)

            x = block1(x, t, spatial_film=sp_film_b1)
            x = block2(x, t, spatial_film=sp_film_b2)
            x = spatial_attn(x)
            x = temporal_attn(
                x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
            )
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        out = self.out_conv(x)
        return out