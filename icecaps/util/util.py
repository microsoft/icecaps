import random
import shutil
import stat
import os
import numpy as np
import tensorflow as tf


def init_rng(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def clean_model_dir(model_dir):
    if os.path.isdir(model_dir):
        print('Cleaning model dir...')
        os.chmod(model_dir, stat.S_IWUSR)
        shutil.rmtree(model_dir, ignore_errors=True)

def load_params(params_file):
    def convert_type(x):
        if x == "True":
            return True
        if x == "False":
            return False
        try:
            if '.' in x or 'e' in x or 'E' in x:
                return float(x)
            else:
                return int(x)
        except ValueError:
            return x
    params = dict()
    with open(params_file, 'r', encoding="utf8") as params_f:
        for line in params_f:
            key, value = line.split(': ')
            params[key.strip()] = convert_type(value.strip())
    return params

def get_gpu_mem_config(dynamic_mem):
    # Configuration for GPU memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = dynamic_mem
    return config
