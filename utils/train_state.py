###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# Interpolate from model to target_model. Tau = ratio of current model to target model
def target_update(model, target_model, tau):
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)

# Contains model params and optimizer state.
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def: nn.Module, params, tx=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model_def.apply, model_def=model_def, params=params,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    # Call model_def.apply_fn.
    def __call__(self, *args, params=None, method=None, **kwargs,):
        if params is None:
            params = self.params
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    # Shortcut for above. Method should be a string.
    def do(self, method):
        return functools.partial(self, method=method)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)

            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'opt_state': self.opt_state,
            'step': self.step,
        }
    
    def load(self, data):
        return self.replace(**data)