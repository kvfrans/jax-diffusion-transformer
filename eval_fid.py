from localutils.debugger import enable_debug
enable_debug()

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt

from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from schedulers import GaussianDiffusion
from diffusion_transformer import DiT
from train_diffusion import DiffusionTrainer
from utils.fid import get_fid_network, fid_from_stats

delattr(flags.FLAGS, 'dataset_name')
delattr(flags.FLAGS, 'load_dir')
delattr(flags.FLAGS, 'batch_size')

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet128', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Load dir (if not None, load params from here).')
flags.DEFINE_string('fid_stats', 'data/imagenet256_fidstats_openai.npz', 'location of fid stats to compare to.')
flags.DEFINE_float('cfg_weight', 4, 'CFG weighting.')
flags.DEFINE_integer('use_cfg', 1, 'Use CFG.')
flags.DEFINE_integer('batch_size', 128, 'Total Batch size.')
flags.DEFINE_integer('diffusion_timesteps', 500, 'Number of diffusion timesteps.')
flags.DEFINE_integer('num_samples', 50000, 'Total samples to generate for FID.')

##############################################
## Training Code.
##############################################
def main(_):
    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    example_obs = jnp.zeros((local_batch_size, 256, 256, 3))
    example_labels = jnp.zeros((local_batch_size), dtype=jnp.int32)

    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_decode_pmap = jax.pmap(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    ###################################
    # Creating Model and put on devices.
    ###################################

    assert FLAGS.load_dir is not None
    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        cp_dict = cp.load_as_dict()
        if 'config' in cp_dict:
            FLAGS.model = cp_dict['config']

    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size = example_obs.shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
    }
    model_def = DiT(**dit_args)
    FLAGS.model['diffusion_timesteps'] = FLAGS.diffusion_timesteps
    
    example_t = jnp.zeros((local_batch_size,))
    example_label = jnp.zeros((local_batch_size,), dtype=jnp.int32)
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    scheduler = GaussianDiffusion(FLAGS.model['diffusion_timesteps'])
    model = DiffusionTrainer(rng, model_ts, model_ts_eps, FLAGS.model, scheduler)

    model = cp.load_model(model)
    print("Loaded model with step", model.model.step)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    ###################################
    ### Generate images for FID use
    ###################################

    def get_dataset(is_train):
        if 'imagenet' in FLAGS.dataset_name:
            def deserialization_fn(data):
                image = data['image']
                min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
                image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
                if 'imagenet256' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (256, 256), antialias=True)
                elif 'imagenet128' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (128, 128), antialias=True)
                else:
                    raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
                if is_train:
                    image = tf.image.random_flip_left_right(image)
                image = tf.cast(image, tf.float32) / 255.0
                image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
                return image, data['label']

            split = tfds.split_for_jax_process('train' if (is_train or FLAGS.debug_overfit) else 'validation', drop_remainder=True)
            dataset = tfds.load('imagenet2012', split=split)
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            if FLAGS.debug_overfit:
                dataset = dataset.take(8)
                dataset = dataset.repeat()
                dataset = dataset.batch(local_batch_size)
            else:
                dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
                dataset = dataset.repeat()
                dataset = dataset.batch(local_batch_size)
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)
            dataset = iter(dataset)
            return dataset
        elif 'celebahq' in FLAGS.dataset_name:
            def deserialization_fn(data):
                image = data['image']
                if 'celebahq256' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (256, 256), antialias=True)
                elif 'celebahq128' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (128, 128), antialias=True)
                else:
                    raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
                image = tf.cast(image, tf.float32)
                image = image / 255.0
                image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
                return image,  data['label']

            dataset = tfds.load('celeba256', split='train')
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(local_batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)
            dataset = iter(dataset)
            return dataset
        else:
            raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
        
    dataset = get_dataset(is_train=True)

    example_obs = example_obs.reshape((len(jax.local_devices()), -1, *example_obs.shape[1:])) # [devices, batch//devices, etc..]
    example_labels = example_labels.reshape((len(jax.local_devices()), -1)) # [devices, batch//devices]
    vmap_split = jax.vmap(jax.random.split, in_axes=(0))

    get_fid_activations = get_fid_network()
    truth_fid_stats = np.load(FLAGS.fid_stats)

    activations = []
    key = jax.random.PRNGKey(42 + jax.process_index())

    num_iters = FLAGS.num_samples // FLAGS.batch_size
    

    for i in tqdm.tqdm(range(num_iters)):
        noise_key, iter_key, key = jax.random.split(key, 3)
        x = jax.random.normal(noise_key, example_obs.shape) # [devices, batch//devices, etc..]
        _, labels = next(dataset)
        labels = labels.reshape((len(jax.local_devices()), -1))

        iter_key = flax.jax_utils.replicate(iter_key, devices=jax.local_devices())
        iter_key += jnp.arange(len(jax.local_devices()), dtype=jnp.uint32)[:, None] * 1000
        for ti in range(FLAGS.model['diffusion_timesteps']):
            rng, iter_key = jnp.split(vmap_split(iter_key), 2, axis=-1)
            rng, iter_key = rng[...,0], iter_key[...,0]
            t = jnp.full((x.shape[0], x.shape[1]), FLAGS.model['diffusion_timesteps']-ti) # [devices, batch//devices]
            cfg_weight_array = jnp.full((x.shape[0],), FLAGS.cfg_weight)
            if FLAGS.use_cfg:
                x = model.denoise_step(x, t, labels, rng, cfg_weight_array)
            else:
                x = model.denoise_step_no_cfg(x, t, labels, rng, cfg_weight_array * 0.0)

        if FLAGS.use_stable_vae:
            x = vae_decode_pmap(x)
        x = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
        x = 2 * x - 1
        acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
        acts = jax.pmap(lambda x: jax.lax.all_gather(x, 'i', axis=0), axis_name='i')(acts)[0] # [global_devices, batch//global_devices, 2048]
        acts = np.array(acts)
        activations.append(acts)

    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape((-1, activations.shape[-1]))
    print(activations.shape)
    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)
    fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])

    print("FID:", fid)



if __name__ == '__main__':
    app.run(main)