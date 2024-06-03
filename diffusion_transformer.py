import math
from typing import Any, Tuple
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

# Port of https://github.com/facebookresearch/DiT/blob/main/models.py into jax.

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, embedding_init=nn.initializers.normal(0.02))

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        # x = nn.Dropout(rate=self.dropout_rate)(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        # output = nn.Dropout(rate=self.dropout_rate)(output)
        return output
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.embed_dim, patch_tuple, patch_tuple, use_bias=self.bias, padding="VALID", kernel_init=nn.initializers.xavier_uniform())(x) # (B, P, P, hidden_size)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x
    
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0) # (1, H*W, D)
    
################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    patch_size: int
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, 
                     kernel_init=nn.initializers.constant(0))(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int
    learn_sigma: bool = False

    @nn.compact
    def __call__(self, x, t, y, train=False, force_drop_ids=None):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        print("DiT: Input of shape", x.shape)
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        out_channels = in_channels if not self.learn_sigma else in_channels * 2
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size
        pos_embed = self.param("pos_embed", get_2d_sincos_pos_embed, self.hidden_size, num_patches)
        pos_embed = jax.lax.stop_gradient(pos_embed)
        x = PatchEmbed(self.patch_size, self.hidden_size)(x) # (B, num_patches, hidden_size)
        print("DiT: After patch embed, shape is", x.shape)  
        x = x + pos_embed
        t = TimestepEmbedder(self.hidden_size)(t) # (B, hidden_size)
        y = LabelEmbedder(self.class_dropout_prob, self.num_classes, self.hidden_size)(
            y, train=train, force_drop_ids=force_drop_ids) # (B, hidden_size)
        c = t + y
        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, c)
        x = FinalLayer(self.patch_size, out_channels, self.hidden_size)(x, c) # (B, num_patches, p*p*c)
        x = jnp.reshape(x, (batch_size, num_patches_side, num_patches_side, 
                            self.patch_size, self.patch_size, out_channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, input_size, input_size, out_channels)
        return x
