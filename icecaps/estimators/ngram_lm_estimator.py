import tensorflow as tf
import copy

from tensorflow.contrib.rnn import BasicRNNCell

from icecaps.estimators.rnn_estimator import RNNEstimator
from icecaps.estimators.convolutional_estimator import ConvolutionalEstimator


class NGramCell(BasicRNNCell):

    def __init__(self, hparams):
        super().__init__(hparams.ngram_dim * hparams.token_embed_dim)
        self.hparams = hparams

    def __call__(self, inputs, state):
        new_state = tf.concat(
            [state[:, self.hparams.token_embed_dim:], inputs], -1)
        return state, new_state


class NGramLMEstimator(RNNEstimator):

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.init_inputs()
            self.init_targets()
            self.build_cell()
            self.build_loss()
            if mode == tf.estimator.ModeKeys.PREDICT:
                self.predictions = {
                    "inputs": self.features["inputs"],
                    "outputs": self.reported_loss
                }
                if "metadata" in self.features:
                    self.predictions["metadata"] = self.features["metadata"]
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_optimizer()
                for var in tf.trainable_variables():
                    # Add histograms for trainable variables
                    tf.summary.histogram(var.op.name, var)
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, train_op=self.train_op)
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = dict()
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, eval_metric_ops=self.eval_metric_ops)

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["ngram_dim"] = cls.make_param(5)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        self.features["targets"] = self.features["inputs"]

    def build_cell(self):
        print("Building cell..")
        with tf.variable_scope('cell'):
            self.cell = NGramCell(self.hparams)
            self.token_embeddings = tf.get_variable(name='embedding',
                                                    shape=[self.src_vocab.size(), self.hparams.token_embed_dim])
            self.inputs_dense = tf.nn.embedding_lookup(
                params=self.token_embeddings, ids=self.inputs_sparse)  # [batch_size, time_step, embedding_size]
            sequence_length = tf.ones(
                [self.batch_size], dtype=tf.int32) * self.hparams.max_length
            initial_state = tf.ones(
                [self.batch_size, self.hparams.ngram_dim], dtype=tf.int32) * self.vocab.start_token_id
            initial_state = tf.reshape(tf.nn.embedding_lookup(
                params=self.token_embeddings, ids=initial_state), [-1, self.hparams.ngram_dim * self.hparams.token_embed_dim])
            self.outputs, self.last_state = tf.nn.dynamic_rnn(
                cell=self.cell, inputs=self.inputs_dense,
                sequence_length=sequence_length, initial_state=initial_state,
                time_major=False)  # [batch_size, max_time_step, cell_output_size], [batch_size, cell_output_size]
            conv_params = copy.copy(self.params)
            conv_params["in_dim"] = self.hparams.ngram_dim
            conv_params["out_dim"] = self.tgt_vocab.size()
            conv_params["conv2d"] = False
            self.outputs = tf.reshape(
                self.outputs, [-1, self.hparams.ngram_dim, self.hparams.token_embed_dim])
            conv_nn = ConvolutionalEstimator(
                self.model_dir + "/conv", conv_params, scope=self.scope + "/conv")
            self.logits = conv_nn._model_fn(
                {"inputs": self.outputs}, tf.estimator.ModeKeys.PREDICT, conv_nn.params).predictions["logits"]
            self.logits = tf.reshape(
                self.logits, [self.batch_size, -1, self.tgt_vocab.size()])
