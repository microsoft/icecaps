import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMStateTuple
import copy
import string
from collections import namedtuple


class AbstractIcecapsEstimator(tf.estimator.Estimator):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope=""):
        self.built = None
        self.scope = scope
        if len(self.scope) > 0 and self.scope[0] == '/':
            self.scope = self.scope[1:]
        if not hasattr(self, "reported_loss_name"):
            self.reported_loss_name = self.scope + "/reported_loss"
            if self.reported_loss_name[0] == '/':
                self.reported_loss_name = self.reported_loss_name[1:]
        run_config = tf.estimator.RunConfig()
        params = self.filter_features(params, self.scope)
        if "summary_steps" in params:
            run_config = run_config.replace(
                save_summary_steps=params["summary_steps"])
        if config:
            run_config = run_config.replace(session_config=config)
        super().__init__(
            model_fn=self._model_fn, model_dir=model_dir, params=params, config=run_config)

    def _model_fn(self, features, mode, params):
        raise NotImplementedError(type(self).__name__ + " is abstract.")

    def dummy_inputs(self):
        raise NotImplementedError(type(self).__name__ + " is abstract.")

    def extract_args(self, features, mode, params):
        self.features = copy.copy(features)
        self.features = self.nest_flattened_tensors(self.features)
        for key in self.features:
            if isinstance(self.features[key], list) or isinstance(self.features[key], np.ndarray):
                self.features[key] = tf.convert_to_tensor(self.features[key])
        self.mode = mode
        expected_params = self.construct_expected_params()
        if "use_default_params" not in params:
            raise ValueError(
                "Please explicitly set parameter use_default_params to True or False.")
        if params["use_default_params"]:
            params = self.complete_params(params, expected_params)
        self.check_params(params, expected_params)
        self.hparams = namedtuple("hparams", params.keys())(**params)
        self.keep_prob = 1.0 - \
            self.hparams.dropout_rate if self.mode == tf.estimator.ModeKeys.TRAIN else 1.0
        self.learning_rate = self.hparams.learning_rate

    @classmethod
    def make_param(cls, value):
        Parameter = namedtuple('Parameter', 'cls type default')
        return Parameter(cls=cls.__name__, type=type(value).__name__, default=value)

    @classmethod
    def construct_expected_params(cls):
        expected_params = dict()
        expected_params["optimizer"] = cls.make_param("adam")
        expected_params["learning_rate"] = cls.make_param(0.0001)
        expected_params["momentum"] = cls.make_param(0.0)
        expected_params["max_gradient_norm"] = cls.make_param(5.0)
        expected_params["dropout_rate"] = cls.make_param(0.2)
        return expected_params

    @classmethod
    def list_params(cls, expected_params=None):
        if expected_params is None:
            expected_params = cls.construct_expected_params()
        print("Hyperparameters:")
        for key in expected_params:
            print("--" + key + " (" + str(expected_params[key].type) + ", " + str(
                expected_params[key].default) + ")")

    def complete_params(self, provided_params=None, expected_params=None):
        if provided_params is None:
            provided_params = self.params
        if expected_params is None:
            expected_params = self.construct_expected_params()
        params = dict()
        for key in expected_params:
            if key in provided_params:
                params[key] = provided_params[key]
            elif self.scope + "/" + key in provided_params:
                params[key] = provided_params[self.scope + '/' + key]
            else:
                params[key] = expected_params[key].default
        return params

    def check_params(self, provided_params, expected_params):
        for key in expected_params:
            if key not in provided_params:
                self.list_params(expected_params)
                raise ValueError(
                    "Key " + key + " is missing from hyperparameters.")
            if type(provided_params[key]).__name__ != expected_params[key].type:
                self.list_params(expected_params)
                print("Current key is " + str(key))
                print("Current expected is " + str(expected_params[key]))
                print("Current provided is " + str(provided_params[key]))
                raise ValueError("Key " + key + " expects type " +
                                 expected_params[key].type + " (got " + type(provided_params[key]).__name__ + ").")

    def filter_features(self, features, scope):
        prefix = str(scope) + "/"
        filtered_features = dict()
        for field in features:
            if "/" not in field:
                filtered_features[field] = features[field]
            elif field.startswith(prefix):
                suffix = field[len(prefix):]
                if "/" in suffix:
                    filtered_features[field] = features[field]
                else:
                    filtered_features[suffix] = features[field]
        return filtered_features

    def chain_outputs(self, features):
        predictions = self._model_fn(
            features, tf.estimator.ModeKeys.PREDICT, self.params).predictions
        outputs = dict(features, **predictions)
        flattened = self.flatten_nested_tensors(outputs)
        return flattened

    def flatten_nested_tensors(self, dict_):
        result = dict()
        def flatten_pass(obj, prefix=''):
            if len(prefix) > 0 and prefix[0] == '/':
                prefix = prefix[1:]
            if isinstance(obj, dict):
                for key in obj:
                    flatten_pass(obj[key], prefix + '/' + key)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    flatten_pass(obj[i], prefix + '/list/' + str(i))
            elif isinstance(obj, tuple):
                tuple_type = str(type(obj))
                if tuple_type == "<class 'tuple'>":
                    flatten_pass(list(obj), prefix)
                else:
                    if '.' in tuple_type:
                        tuple_type = tuple_type[tuple_type.rfind('.') + 1:]
                        tuple_type = "".join(
                            (char for char in tuple_type if char not in string.punctuation))
                    flatten_pass(obj._asdict(), prefix +
                                 '/tuple/' + tuple_type)
            else:
                result[prefix] = obj
        flatten_pass(dict_)
        return result

    def nest_flattened_tensors(self, dict_):
        mid_result = dict()

        def in_(key, struct):
            if isinstance(struct, list):
                return key < len(struct) and struct[key] is not None
            else:
                return key in struct
        for key, value in dict_.items():
            tokens = key.split('/')
            current_struct = mid_result
            current_key = ''
            i = 0
            while i < len(tokens):
                if tokens[i] == 'list':
                    if not in_(current_key, current_struct):
                        current_struct[current_key] = []
                    current_struct = current_struct[current_key]
                    i += 1
                    current_key = int(tokens[i])
                    while current_key >= len(current_struct):
                        current_struct.append(None)
                elif tokens[i] == 'tuple':
                    if not in_(current_key, current_struct):
                        i += 1
                        tuple_type = tokens[i]
                        if '.' in tuple_type:
                            tuple_type = tuple_type[tuple_type.rfind('.') + 1:]
                            tuple_type = "".join(
                                (char for char in tuple_type if char not in string.punctuation))
                        current_struct[current_key] = {'name__': tuple_type}
                    current_struct = current_struct[current_key]
                else:
                    current_key = tokens[i]
                i += 1
            current_struct[current_key] = value

        def tuplify_pass(obj, outer_struct=None, outer_key=None):
            if isinstance(obj, dict):
                if 'name__' in obj:
                    tuplified = copy.copy(obj)
                    del tuplified['name__']
                    if obj['name__'] == "LSTMStateTuple":
                        tuplified = LSTMStateTuple(**tuplified)
                    else:
                        tuplified = namedtuple(
                            obj['name__'], sorted(tuplified))(**tuplified)
                    outer_struct[outer_key] = tuplified
                else:
                    for key in obj:
                        tuplify_pass(obj[key], obj, key)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    tuplify_pass(obj[i], obj, i)
            else:
                outer_struct[outer_key] = obj
        tuplify_pass(mid_result)
        result = dict()
        for key, value in mid_result.items():
            if isinstance(value, list):
                result[key] = tuple(value)
            else:
                result[key] = value
        return result

    def get_num_model_params(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def build_optimizer(self, trainable_variables=None):
        with tf.name_scope('optimizer'):
            if self.hparams.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)
            elif self.hparams.optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(
                    self.learning_rate, self.hparams.momentum)
            elif self.hparams.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(
                    self.learning_rate, epsilon=1e-4)
            elif self.hparams.optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(
                    self.learning_rate)
            elif self.hparams.optimizer == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate)
            elif self.hparams.optimizer == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate)
            else:
                raise ValueError("Optimizer " + optimizer + " not recognized. Aborting...")
            with tf.name_scope('gradients'):
                if not trainable_variables:
                    trainable_variables = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_variables)
                clipped, norm = tf.clip_by_global_norm(
                    gradients, self.hparams.max_gradient_norm)
                grads_and_vars = []
                for i in range(len(clipped)):
                    grads_and_vars.append((clipped[i], trainable_variables[i]))
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars, global_step=tf.train.get_global_step())

    def train(self, input_fn, hooks=[], steps=None, max_steps=None, saving_listeners=None, logging_freq=10):
        try:
            print('Training..')
            if steps == 0 or max_steps == 0:
                return
            tensors_to_log = {"loss": self.reported_loss_name}
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=logging_freq)
            super().train(input_fn, hooks=hooks+[logging_hook], steps=steps)
            print('Training terminated.')
        except tf.errors.InvalidArgumentError:
            print('\nModel mismatch!')
            raise

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        try:
            print('Evaluating..')
            if steps == 0:
                return
            results = super().evaluate(input_fn, hooks=hooks)
            print('Evaluating terminated.')
            return results
        except tf.errors.InvalidArgumentError:
            print('\nModel mismatch!')
            raise
