import tensorflow as tf
import copy
from collections import OrderedDict

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator


class AbstractEstimatorGroup(AbstractIcecapsEstimator):

    def __init__(self, estimator_ls, model_dir="/tmp", params=dict(), config=None, scope=""):
        self.estimator_map = OrderedDict()
        self.balance_map = OrderedDict()
        for estimator in estimator_ls:
            self.estimator_map[estimator.scope] = estimator
            self.balance_map[estimator.scope] = 0
        self.reported_loss_name = estimator_ls[0].reported_loss_name
        super().__init__(model_dir, params, config, scope)

    def _model_fn(self, features, mode, params):
        self.extract_args(features, mode, params)
        def get_model_fn(scope):
            filtered_features = self.filter_features(features, scope)
            return self.estimator_map[scope]._model_fn(filtered_features, mode, self.estimator_map[scope].params)
        if mode != tf.estimator.ModeKeys.TRAIN:
            return get_model_fn(max(self.balance_map, key=lambda key: self.balance_map[key]))
        for key in self.estimator_map:
            _ = get_model_fn(key)
        train_op = tf.no_op()
        with tf.control_dependencies([train_op]):
            for key in self.balance_map:
                for _ in range(self.balance_map[key]):
                    train_op = tf.group(
                        [train_op, self.estimator_map[key].train_op])
        self.loss_map = {key: get_model_fn(key).loss for key in self.balance_map}
        loss = None
        for key in self.balance_map:
            if loss is None:
                loss = self.balance_map[key] * self.loss_map[key]
            else:
                loss += self.balance_map[key] * self.loss_map[key]
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    def set_spec(self, spec):
        if isinstance(spec, str):
            for key in self.balance_map:
                self.balance_map[key] = int(key == spec)
        elif isinstance(spec, AbstractIcecapsEstimator):
            for key in self.balance_map:
                self.balance_map[key] = int(self.estimator_map[key] == spec)
        elif isinstance(spec, dict) or isinstance(spec, OrderedDict):
            for key in self.balance_map:
                self.balance_map[key] = spec[key]

    def train(self, _input_fn, spec=None, hooks=[], steps=None, max_steps=None, saving_listeners=None, logging_freq=100):
        if spec is None:
            spec = {x: 1 for x in self.estimator_map}
        self.set_spec(spec)
        return super().train(_input_fn, hooks, steps, max_steps, saving_listeners, logging_freq)

    def evaluate(self, _input_fn, spec=None, steps=None, hooks=None, checkpoint_path=None, name=None):
        if spec is None:
            spec = {x: 1 for x in self.estimator_map}
        self.set_spec(spec)
        return super().evaluate(_input_fn, steps, hooks, checkpoint_path, name)

    def predict(self, _input_fn, spec=None, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        if spec is None:
            spec = {x: 1 for x in self.estimator_map}
        self.set_spec(spec)
        return super().predict(_input_fn, predict_keys, hooks, checkpoint_path, yield_single_examples)


class EstimatorGroup(AbstractEstimatorGroup):
    pass
    