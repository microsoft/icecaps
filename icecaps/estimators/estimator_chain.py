import tensorflow as tf
import copy
from collections import namedtuple

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator


class EstimatorChain(AbstractIcecapsEstimator):

    def __init__(self, process_ls, model_dir="/tmp", params=dict(), config=None, scope=""):
        self.process_ls = process_ls
        if isinstance(self.process_ls[-1], AbstractIcecapsEstimator):
            self.reported_loss_name = self.process_ls[-1].reported_loss_name
        else:
            raise ValueError("Last component must be an estimator.")
        super().__init__(model_dir, params, config, scope)

    def calculate_subchain(self, features, mode, params, length=0):
        self.extract_args(features, mode, params)
        if length <= 0:
            length += len(self.process_ls)
        if (not isinstance(self.process_ls[length - 1], AbstractIcecapsEstimator)) and mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError(
                "Cannot train or evaluate a subchain where the last component is not an estimator.")
        for i in range(length):
            if isinstance(self.process_ls[i], AbstractIcecapsEstimator):
                estimator = self.process_ls[i]
                if i + 1 == length:
                    if mode == tf.estimator.ModeKeys.PREDICT and length < len(self.process_ls):
                        predictions = estimator.chain_outputs(self.features)
                        output = tf.estimator.EstimatorSpec(mode, predictions=predictions)
                    else:
                        output = estimator._model_fn(
                            self.features, mode, estimator.params)
                        if mode == tf.estimator.ModeKeys.TRAIN:
                            self.train_op = estimator.train_op
                    return output
                else:
                    predictions = estimator.chain_outputs(self.features)
                    for field in predictions:
                        if field == "outputs":
                            self.features["inputs"] = predictions[field]
                        elif field == "inputs":
                            self.features["original_inputs"] = predictions[field]
                        else:
                            self.features[field] = predictions[field]
            else:
                self.features = self.process_ls[i](self.features)
                if i + 1 == length:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=self.features)

    def _model_fn(self, features, mode, params):
        return self.calculate_subchain(features, mode, params, length=0)

