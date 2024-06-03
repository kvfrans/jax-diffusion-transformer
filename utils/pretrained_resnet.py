"""Pretrained image models for use during training. 
https://github.com/google-research/magvit/blob/main/videogvt/train_lib/pretrained_model_utils.py"""

from typing import Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import functools
from typing import Any, Callable, Optional, Tuple, Type, Union, Sequence, List

# Download from: gsutil cp gs://gresearch/xmcgan/resnet_pretrained.npy data/
_DEFAULT_RESNET_PATH = None

RESNET_IMG_SIZE = 224
VALID_MODELS = ["resnet50"]

@flax.struct.dataclass
class ModelState:
    params: flax.core.FrozenDict
    batch_stats: flax.core.FrozenDict

class ObjectFromDict(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a,[ObjectFromDict(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, ObjectFromDict(b) if isinstance(b, dict) else b)


def create_train_state(config: ml_collections.ConfigDict, rng: np.ndarray,
                       input_shape: Sequence[int],
                       num_classes: int) -> Tuple[nn.Module, ModelState]:
    """Create and initialize the model.

    Args:
        config: Configuration for model.
        rng: JAX PRNG Key.
        input_shape: Shape of the inputs fed into the model.
        num_classes: Number of classes in the output layer.

    Returns:
        model: Flax nn.Nodule model architecture.
        state: The initialized ModelState with the optimizer.
    """
    if config.model_name == "resnet50":
        model_cls = ResNet50
    else:
        raise ValueError(f"Model {config.model_name} not supported.")
    model = model_cls(num_classes=num_classes)
    variables = model.init(rng, jnp.ones(input_shape), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    return model, ModelState(params=params, batch_stats=batch_stats)


def get_pretrained_model(
    model_name: str = "resnet50",
    checkpoint_path: Optional[str] = _DEFAULT_RESNET_PATH
) -> Tuple[nn.Module, ModelState]:
    """Returns a pretrained model loaded from weights in checkpoint_dir.

    Args:
        model_name: Name of model architecture to load. Currently only supports
        "resnet50".
        checkpoint_path: Path of .npy containing pretrained state.

    Returns:
        model: Flax nn.Nodule model architecture.
        state: The initialized ModelState with the optimizer.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"Model {model_name} not supported.")
    # Initialize model.
    config = ml_collections.ConfigDict()
    config.model_name = "resnet50"
    config.sgd_momentum = 0.9  # Unused for inference.
    config.seed = 42  # Unused for inference.
    model_rng = jax.random.PRNGKey(config.seed)
    model, state = create_train_state(
        config,
        model_rng,
        input_shape=(1, RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3),
        num_classes=1000)

    if checkpoint_path is not None:
        # Set up checkpointing of the model and the input pipeline.'
        tf.io.gfile.makedirs(checkpoint_path.rsplit("/", 1)[0])
        with tf.io.gfile.GFile(checkpoint_path, "rb") as f:
            checkpoint_data = np.load(f, allow_pickle=True).item()
            state = ModelState(
                params=checkpoint_data["params"],
                batch_stats=checkpoint_data["batch_stats"])
    return model, state


def get_pretrained_embs(state: ModelState, model: nn.Module,
                        images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract embeddings from a pretrained model.

    Args:
        state: ModelState containing model parameters.
        model: Pretrained Flax model.
        images: Array of shape (H, W, 3).

    Returns:
        pool: Pooled outputs from intermediate layer of shape (H', W', C).
        outputs: Outputs from last layer with shape (num_classes,).
    """

    if len(images.shape) != 4 or images.shape[3] != 3:
        raise ValueError("images should be of shape (H, W, 3).")
    if images.shape[1] != RESNET_IMG_SIZE and images.shape[2] != RESNET_IMG_SIZE:
        images = jax.image.resize(
            images,
            (images.shape[0], RESNET_IMG_SIZE, RESNET_IMG_SIZE, images.shape[3]),
            "bilinear")
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    pool, outputs = model.apply(variables, images, mutable=False, train=False)
    return pool, outputs


# ==========================
# ==========================
# ==========================

Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), use_bias=False)
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), use_bias=False)


class ResNetBlock(nn.Module):
    """ResNet block without bottleneck used in ResNet-18 and ResNet-34."""

    filters: int
    norm: Any
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x

        x = Conv3x3(self.filters, strides=self.strides, name="conv1")(x)
        x = self.norm(name="bn1")(x)
        x = nn.relu(x)
        x = Conv3x3(self.filters, name="conv2")(x)
        # Initializing the scale to 0 has been common practice since "Fixup
        # Initialization: Residual Learning Without Normalization" Tengyu et al,
        # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
        x = self.norm(scale_init=nn.initializers.zeros, name="bn2")(x)

        if residual.shape != x.shape:
            residual = Conv1x1(
                self.filters, strides=self.strides, name="proj_conv")(
                    residual)
            residual = self.norm(name="proj_bn")(residual)

        x = nn.relu(residual + x)
        return x


class BottleneckResNetBlock(ResNetBlock):
    """Bottleneck ResNet block used in ResNet-50 and larger."""

    @nn.compact
    def __call__(self, x):
        residual = x

        x = Conv1x1(self.filters, name="conv1")(x)
        x = self.norm(name="bn1")(x)
        x = nn.relu(x)
        x = Conv3x3(self.filters, strides=self.strides, name="conv2")(x)
        x = self.norm(name="bn2")(x)
        x = nn.relu(x)
        x = Conv1x1(4 * self.filters, name="conv3")(x)
        # Initializing the scale to 0 has been common practice since "Fixup
        # Initialization: Residual Learning Without Normalization" Tengyu et al,
        # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
        x = self.norm(name="bn3")(x)

        if residual.shape != x.shape:
            residual = Conv1x1(
                4 * self.filters, strides=self.strides, name="proj_conv")(
                    residual)
            residual = self.norm(name="proj_bn")(residual)

        x = nn.relu(residual + x)
        return x


class ResNetStage(nn.Module):
    """ResNet stage consistent of multiple ResNet blocks."""

    stage_size: int
    filters: int
    block_cls: Type[ResNetBlock]
    norm: Any
    first_block_strides: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        for i in range(self.stage_size):
            x = self.block_cls(
                filters=self.filters,
                norm=self.norm,
                strides=self.first_block_strides if i == 0 else (1, 1),
                name=f"block{i + 1}")(x)
        return x


class ResNet(nn.Module):
    """Construct ResNet V1 with `num_classes` outputs.

    Attributes:
        num_classes: Number of nodes in the final layer.
        block_cls: Class for the blocks. ResNet-50 and larger use
        `BottleneckResNetBlock` (convolutions: 1x1, 3x3, 1x1), ResNet-18 and
            ResNet-34 use `ResNetBlock` without bottleneck (two 3x3 convolutions).
        stage_sizes: List with the number of ResNet blocks in each stage. Number of
        stages can be varied.
        width_factor: Factor applied to the number of filters. The 64 * width_factor
        is the number of filters in the first stage, every consecutive stage
        doubles the number of filters.
    """
    num_classes: int
    block_cls: Type[ResNetBlock]
    stage_sizes: List[int]
    width_factor: int = 1

    @nn.compact
    def __call__(self, x, *, train: bool):
        """Apply the ResNet to the inputs `x`.

        Args:
        x: Inputs.
        train: Whether to use BatchNorm in training or inference mode.

        Returns:
        pool: The intermediate output of shape (N, 7, 7, 2048).
        out: The output head with shape (N, num_classes).
        """
        width = 64 * self.width_factor
        norm = functools.partial(
            nn.BatchNorm, use_running_average=not train, momentum=0.9)

        # Root block
        x = nn.Conv(
            features=width,
            kernel_size=(7, 7),
            strides=(2, 2),
            use_bias=False,
            name="init_conv")(
                x)
        x = norm(name="init_bn")(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # Stages
        for i, stage_size in enumerate(self.stage_sizes):
            x = ResNetStage(
                stage_size,
                filters=width * 2**i,
                block_cls=self.block_cls,
                norm=norm,
                first_block_strides=(1, 1) if i == 0 else (2, 2),
                name=f"stage{i + 1}")(
                    x)

        # Head
        pool = x
        out = jnp.mean(pool, axis=(1, 2))
        out = nn.Dense(
            self.num_classes, kernel_init=nn.initializers.zeros, name="head")(out)
        return pool, out


ResNet18 = functools.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = functools.partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = functools.partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = functools.partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)