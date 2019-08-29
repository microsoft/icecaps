import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from icecaps.estimators.abstract_recurrent_estimator import AbstractRecurrentEstimator
from icecaps.util.vocabulary import Vocabulary


class Seq2SeqEncoderEstimator(AbstractRecurrentEstimator):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope=""):
        super().__init__(model_dir, params, config, scope)

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.build_rnn("inputs")
            if mode == tf.estimator.ModeKeys.PREDICT:
                self.predictions = self.flatten_nested_tensors({
                    "inputs": self.features["inputs"],
                    "outputs": self.outputs,
                    "state": self.last_state,
                    "length": self.inputs_length,
                    "token_embeddings": tf.identity(self.token_embeddings),
                })
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            if mode == tf.estimator.ModeKeys.TRAIN:
                raise NotImplementedError(
                    "Training not currently supported for seq2seq encoder.")
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = dict()
                raise NotImplementedError(
                    "Evaluation not currently supported for seq2seq encoder.")

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)

    def build_loss(self):
        raise NotImplementedError("No loss function for seq2seq encoder.")
