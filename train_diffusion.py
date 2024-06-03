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

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from schedulers import GaussianDiffusion
from diffusion_transformer import DiT

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')

flags.DEFINE_integer('debug_overfit', 0, 'For debug, overfit to a single batch.')
flags.DEFINE_integer('use_stable_vae', 1, 'Use stable vae.')

model_config = ml_collections.ConfigDict({
    # Make sure to run with Large configs when we actually want to run!
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.99,
    'diffusion_timesteps': 500,
    'hidden_size': 64,
    'patch_size': 8,
    'depth': 2,
    'num_heads': 2,
    'mlp_ratio': 1,
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'cfg_scale': 4.0,
    'eps_update_rate': 0.9999,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'diffusion',
    'name': 'diffusion_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Model Definitions.
##############################################

class DiffusionTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    scheduler: Any = flax.struct.field(pytree_node=False)

    @partial(jax.pmap, axis_name='data')
    def update(self, images, labels, pmap_axis='data'):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            random_t = jax.random.randint(time_key, (images.shape[0],), 0, self.config['diffusion_timesteps'])
            eps = jax.random.normal(noise_key, images.shape)
            x_t = self.scheduler.q_sample(images, random_t, eps)
            
            eps_prime = self.model(x_t, random_t, labels, train=True, rngs={'label_dropout': label_key}, params=params)
            l2_loss_eps = jnp.mean((eps_prime - eps) ** 2)
            
            loss = l2_loss_eps
            return loss, {
                'l2_loss_eps': l2_loss_eps,
                'eps_abs_mean': jnp.abs(eps).mean(),
                'eps_pred_abs_mean': jnp.abs(eps_prime).mean(),
            }
        
        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        updates, new_opt_state = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(step=self.model.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)

        new_model_eps = target_update(self.model, self.model_eps, 1-self.config['eps_update_rate'])

        new_trainer = self.replace(rng=new_rng, model=new_model, model_eps=new_model_eps)
        return new_trainer, info
    
    @partial(jax.jit, static_argnames=('cfg'))
    def call_model(self, images, t, labels, cfg=True, cfg_val=1.0):
        if not cfg:
            return self.model_eps(images, t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * self.config['num_classes'] # Null token
            images_expanded = jnp.tile(images, (2, 1, 1, 1)) # (batch*2, h, w, c)
            t_expanded = jnp.tile(t, (2,)) # (batch*2,)
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            eps_pred = self.model_eps(images_expanded, t_expanded, labels_full, train=False, force_drop_ids=False)
            eps_label = eps_pred[:images.shape[0]]
            eps_uncond = eps_pred[images.shape[0]:]
            eps = eps_uncond + cfg_val * (eps_label - eps_uncond)
            return eps
    
    @partial(jax.pmap, axis_name='data', static_broadcasted_argnums=(4,5))
    def call_model_pmap(self, images, t, labels, cfg=True, cfg_val=1.0):
        return self.call_model(images, t, labels, cfg=cfg, cfg_val=cfg_val)
    
    @partial(jax.pmap, axis_name='data')
    def validation_denoise(self, images, labels, t, pmap_axis='data'):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)
        eps = jax.random.normal(noise_key, images.shape)
        x_t = self.scheduler.q_sample(images, t, eps)
        eps_prime = self.call_model(x_t, t, labels, cfg=True, cfg_val=self.config['cfg_scale'])

        pred_x0 = self.scheduler._predict_xstart_from_eps(x_t, t, eps_prime)
        return x_t, pred_x0, eps_prime
    
    @partial(jax.pmap, axis_name='data')
    def denoise_step(self, x, t, labels, rng, cfg_val, pmap_axis='data'):
        t = jnp.full((x.shape[0],), t)
        eps_prime = self.call_model(x, t, labels, cfg=True, cfg_val=cfg_val)
        mean, variance, log_variance = self.scheduler.p_mean_variance(x, t, eps_prime, clip=False)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(rng, x.shape)
        return x
    
    @partial(jax.pmap, axis_name='data')
    def denoise_step_no_cfg(self, x, t, labels, rng, cfg_val, pmap_axis='data'):
        t = jnp.full((x.shape[0],), t)
        eps_prime = self.call_model(x, t, labels, cfg=False, cfg_val=cfg_val)
        mean, variance, log_variance = self.scheduler.p_mean_variance(x, t, eps_prime, clip=False)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(rng, x.shape)
        return x

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

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

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

            dataset = tfds.load('celeba256', split='train', data_dir='gs://rll-tpus-kvfrans/tfds')
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
    dataset_valid = get_dataset(is_train=False)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:local_batch_size]

    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode_pmap = jax.pmap(vae.encode)
        vae_decode = jax.jit(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    ###################################
    # Creating Model and put on devices.
    ###################################
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
    
    example_t = jnp.zeros((local_batch_size,))
    example_label = jnp.zeros((local_batch_size,), dtype=jnp.int32)
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    scheduler = GaussianDiffusion(FLAGS.model['diffusion_timesteps'])
    model = DiffusionTrainer(rng, model_ts, model_ts_eps, FLAGS.model, scheduler)

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    valid_images_small, valid_labels_small = next(dataset_valid)
    valid_images_small = valid_images_small[:8]
    valid_images_small = valid_images_small.reshape((device_count, -1, *valid_images_small.shape[1:]))
    valid_labels_small = valid_labels_small[:8]
    valid_labels_small = valid_labels_small.reshape((device_count, -1, *valid_labels_small.shape[1:]))
    visualize_labels = example_labels.reshape((device_count, -1, *example_labels.shape[1:]))
    visualize_labels = visualize_labels[:, 0:1]
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()
    if FLAGS.use_stable_vae:
        valid_images_small = vae_encode_pmap(vae_rng, valid_images_small)

    ###################################
    # Train Loop
    ###################################
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = next(dataset)
            batch_images = batch_images.reshape((len(jax.local_devices()), -1, *batch_images.shape[1:])) # [devices, batch//devices, etc..]
            batch_labels = batch_labels.reshape((len(jax.local_devices()), -1, *batch_labels.shape[1:]))
            if FLAGS.use_stable_vae:
                batch_images = vae_encode_pmap(vae_rng, batch_images)

        model, update_info = model.update(batch_images, batch_labels)

        if i % FLAGS.log_interval == 0:
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0 or i == 1000:
            # Validation Losses
            valid_images, valid_labels = next(dataset_valid)
            valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
            valid_labels = valid_labels.reshape((len(jax.local_devices()), -1, *valid_labels.shape[1:]))
            if FLAGS.use_stable_vae:
                valid_images = vae_encode_pmap(vae_rng, valid_images)
            _, valid_update_info = model.update(valid_images, valid_labels)
            valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
            valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
            if jax.process_index() == 0:
                wandb.log(valid_metrics, step=i)

            def process_img(img):
                if FLAGS.use_stable_vae:
                    img = vae_decode(img[None])[0]
                img = img * 0.5 + 0.5
                img = jnp.clip(img, 0, 1)
                return img

            # Training loss on various t.
            mse_total = []
            for t in np.arange(0, 11):
                key = jax.random.PRNGKey(42)
                t_scaled = (FLAGS.model['diffusion_timesteps'] // 10) * t
                t_full = jnp.full((batch_images.shape[0], batch_images.shape[1]), t_scaled)
                eps = jax.random.normal(key, batch_images.shape)
                x_t = model.scheduler.q_sample(batch_images, t_full, eps)
                pred_eps = model.call_model_pmap(x_t, t_full, batch_labels, False, 0.0)
                mse_loss = jnp.mean((eps - pred_eps) ** 2)
                mse_total.append(mse_loss)
                if jax.process_index() == 0:
                    wandb.log({f'training_loss_t/{t}': mse_loss}, step=i)
            mse_total = jnp.array(mse_total[1:-1])
            if jax.process_index() == 0:
                wandb.log({'training_loss_t/mean': mse_total.mean()}, step=i)

            # Validation loss on various t.
            mse_total = []
            for t in np.arange(0, 11):
                key = jax.random.PRNGKey(42)
                t_scaled = (FLAGS.model['diffusion_timesteps'] // 10) * t
                t_full = jnp.full((valid_images.shape[0], valid_images.shape[1]), t_scaled)
                eps = jax.random.normal(key, valid_images.shape)
                x_t = model.scheduler.q_sample(valid_images, t_full, eps)
                pred_eps = model.call_model_pmap(x_t, t_full, valid_labels, True, FLAGS.model['cfg_scale'])
                pred_x0 = model.scheduler._predict_xstart_from_eps(x_t, t_full, pred_eps)
                mse_loss = jnp.mean((eps - pred_eps) ** 2)
                mse_total.append(mse_loss)
                if jax.process_index() == 0:
                    wandb.log({f'validation_loss_t/{t}': mse_loss}, step=i)
            mse_total = jnp.array(mse_total[1:-1])
            if jax.process_index() == 0:
                wandb.log({'validation_loss_t/mean': mse_total.mean()}, step=i)

            # Noise valid images halfway, then denoise.
            assert valid_images_small.shape[0] == len(jax.local_devices()) # [devices, batch//devices, etc..]
            print(valid_images_small.shape)
            print(valid_labels_small.shape)
            t = jnp.arange(valid_images_small.shape[0]) * (FLAGS.model['diffusion_timesteps'] // (valid_images_small.shape[0]))
            t = jnp.tile(t[:, None], (1, valid_images_small.shape[1]))
            x_t, pred_x0, _ = model.validation_denoise(valid_images_small, valid_labels_small, t)
            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(3, 8, figsize=(30, 20))
                for j in range(8):
                    axs[0, j].imshow(process_img(valid_images_small[j, 0]), vmin=0, vmax=1)
                    axs[1, j].imshow(process_img(x_t[j, 0]), vmin=0, vmax=1)
                    axs[2, j].imshow(process_img(pred_x0[j, 0]), vmin=0, vmax=1)
                    axs[0, j].axis('off')
                    axs[1, j].axis('off')
                    axs[2, j].axis('off')
                    # label with the label.
                    axs[0, j].set_title(f"{imagenet_labels[valid_labels_small[j, 0]]}")
                wandb.log({'reconstruction': wandb.Image(fig)}, step=i)
                plt.close(fig)

            # Full Denoising;
            for cfg_weight in [0, 0.1, 1, 4, 10]:
                key = jax.random.PRNGKey(42 + jax.process_index() + i)
                x = jax.random.normal(key, valid_images_small.shape) # [devices, batch//devices, etc..]
                key = flax.jax_utils.replicate(key, devices=jax.local_devices())
                key += jnp.arange(len(jax.local_devices()), dtype=jnp.uint32)[:, None] * 1000
                vmap_split = jax.vmap(jax.random.split, in_axes=(0))
                all_x = []
                for ti in range(FLAGS.model['diffusion_timesteps']):
                    rng, key = jnp.split(vmap_split(key), 2, axis=-1)
                    rng, key = rng[...,0], key[...,0]
                    t = jnp.full((x.shape[0], x.shape[1]), FLAGS.model['diffusion_timesteps']-ti) # [devices, batch//devices]
                    cfg_weight_array = jnp.full((x.shape[0],), cfg_weight)
                    x = model.denoise_step(x, t, visualize_labels, rng, cfg_weight_array)
                    if ti % (FLAGS.model['diffusion_timesteps'] // 16) == 0 or ti == FLAGS.model['diffusion_timesteps']-1:
                        all_x.append(np.array(x))
                all_x = np.stack(all_x, axis=2) # [devices, batch//devices, timesteps, etc..]
                all_x = all_x[:, :, -16:]
                all_x_flat = all_x.reshape((-1, *all_x.shape[2:]))

                if jax.process_index() == 0:
                    # plot comparison witah matplotlib. put each reconstruction side by side.
                    fig, axs = plt.subplots(16, 8, figsize=(30, 60))
                    for j in range(8):
                        for t in range(16):
                            axs[t, j].imshow(process_img(all_x_flat[j, t]), vmin=0, vmax=1)
                        axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
                    wandb.log({f'sample/cfg_{cfg_weight}': wandb.Image(fig)}, step=i)
                    plt.close(fig)
            
            del valid_images, valid_labels
            del all_x, x, x_t, pred_x0, pred_eps, eps


        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(FLAGS.save_dir, parallel=False)
                cp.set_model(model_single)
                cp.save()
                del cp, model_single

if __name__ == '__main__':
    app.run(main)