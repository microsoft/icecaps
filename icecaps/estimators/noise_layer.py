import tensorflow as tf

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator


class NoiseLayer(AbstractIcecapsEstimator):

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            if mode == tf.estimator.ModeKeys.PREDICT:
                noise = tf.random_normal(shape=tf.shape(
                    self.features["inputs"]), stddev=self.hparams.stddev)
                self.outputs = self.features["inputs"] + noise
                self.predictions = {
                    "inputs": self.features["inputs"],
                    "outputs": self.outputs,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            if mode == tf.estimator.ModeKeys.TRAIN:
                raise NotImplementedError(
                    "Training not supported for this estimator.")
            if mode == tf.estimator.ModeKeys.EVAL:
                raise NotImplementedError(
                    "Evaluation not supported for this estimator.")

    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["stddev"] = cls.make_param(0.1)
        return expected_params
