import concurrent.futures
import pickle
import time
import os
import shutil

############
# A checkpointer class for flax models, compatible with saving/loading from gs:// buckets.
# Taken from: https://github.com/danijar/elements/blob/main/elements/checkpoint.py
############

def parent_dir(filename):
    return filename.rsplit('/', 1)[0]

def name(filename):
    return filename.rsplit('/', 1)[1]

class Checkpoint:
    def __init__(self, filename, parallel=True):
        self._filename = filename
        self._values = {}
        self._parallel = parallel
        if self._parallel:
            self._worker = concurrent.futures.ThreadPoolExecutor(1, 'checkpoint')
            self._promise = None

    def __setattr__(self, name, value):
        if name in ('exists', 'save', 'load'):
            return super().__setattr__(name, value)
        if name.startswith('_'):
            return super().__setattr__(name, value)
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except AttributeError:
            raise ValueError(name)
        
    def set_model(self, model):
        for key in model.__dict__.keys():
            data = getattr(model, key)
            if hasattr(data, 'save') or key == 'config':
                self._values[key] = getattr(model, key)

    def save(self, filename=None, keys=None):
        assert self._filename or filename
        filename = filename or self._filename
        print(f'Writing checkpoint: {filename}')
        if self._parallel:
            self._promise and self._promise.result()
            self._promise = self._worker.submit(self._save, filename, keys)
        else:
            self._save(filename, keys)

    def _save(self, filename, keys):
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith('_') for k in keys]), keys
        data = {k: (self._values[k].save() if k != 'config' else self._values[k]) for k in keys}
        data['_timestamp'] = time.time()
        content = pickle.dumps(data)
        if 'gs://' in filename:
            import tensorflow as tf
            tf.io.gfile.makedirs(parent_dir(filename))
            with tf.io.gfile.GFile(filename, 'wb') as f:
                f.write(content)
        else:
            os.makedirs(filename, exist_ok=True)
            tmp = parent_dir(filename) + '/' + name(filename) + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(content)
            shutil.move(tmp, filename)
        print('Wrote checkpoint.')

    def load_as_dict(self, filename=None):
        assert self._filename or filename
        filename = filename or self._filename
        if 'gs://' in filename:
            import tensorflow as tf
            with tf.io.gfile.GFile(filename, 'rb') as f:
                data = pickle.loads(f.read())
        else:
            with open(filename, 'rb') as f:
                data = pickle.loads(f.read())
        age = time.time() - data['_timestamp']
        print(f'Loaded checkpoint from {age:.0f} seconds ago.')
        return data
    
    def load_model(self, model, filename=None):
        cp_dict = self.load_as_dict()
        replace_dict = {}
        for key in model.__dict__.keys():
            if key in cp_dict and key != 'config':
                replace_dict[key] = getattr(model, key).load(cp_dict[key])
        return model.replace(**replace_dict)
