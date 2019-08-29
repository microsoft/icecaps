import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense

from icecaps.estimators.abstract_recurrent_estimator import AbstractRecurrentEstimator
from icecaps.util.vocabulary import Vocabulary


class RNNEstimator(AbstractRecurrentEstimator):

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.init_inputs()
            self.build_cell()
            self.build_obj()
            if mode == tf.estimator.ModeKeys.PREDICT:
                self.build_rt_decoder()
                self.predictions = {
                    "inputs": self.features["inputs"],
                    "outputs": self.hypotheses,
                    "scores": self.scores
                }
                if "metadata" in self.features:
                    self.predictions["metadata"] = self.features["metadata"]
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            self.init_targets()
            self.build_loss()
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
        expected_params["src_vocab_file"] = cls.make_param("")
        expected_params["tgt_vocab_file"] = cls.make_param("")
        expected_params["src_vocab_size"] = cls.make_param(0)
        expected_params["tgt_vocab_size"] = cls.make_param(0)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        if (self.hparams.src_vocab_size == 0 and self.hparams.tgt_vocab_size == 0 and
                self.hparams.src_vocab_file == "" and self.hparams.tgt_vocab_file == ""):
            self.src_vocab = self.vocab
            self.tgt_vocab = self.vocab
        else:
            if self.hparams.src_vocab_size > 0:
                self.src_vocab = Vocabulary(size=self.hparams.src_vocab_size)
            else:
                self.src_vocab = Vocabulary(fname=self.hparams.src_vocab_file)
            if self.hparams.tgt_vocab_size > 0:
                self.tgt_vocab = Vocabulary(size=self.hparams.tgt_vocab_size)
            else:
                self.tgt_vocab = Vocabulary(fname=self.hparams.tgt_vocab_file)

    def init_inputs(self):
        with tf.name_scope('init_encoder'):
            inputs = tf.cast(self.features["inputs"], tf.int32)
            self.batch_size = tf.shape(inputs)[0]
            inputs_length = tf.cast(tf.count_nonzero(
                inputs - self.vocab.end_token_id, -1), tf.int32)
            inputs_max_length = tf.reduce_max(inputs_length)
            end_token = tf.ones(
                shape=[self.batch_size, self.hparams.max_length - inputs_max_length], dtype=tf.int32) * self.vocab.end_token_id
            # [batch_size, max_time_steps + 1]
            self.inputs_sparse = tf.concat([inputs, end_token], axis=1)

    def init_targets(self):
        with tf.name_scope('init_decoder'):
            targets = tf.cast(self.features["targets"], tf.int32)
            targets_length = tf.cast(tf.count_nonzero(
                targets - self.vocab.end_token_id, -1), tf.int32)
            targets_max_length = tf.reduce_max(targets_length)
            end_token = tf.ones(
                shape=[self.batch_size, self.hparams.max_length - targets_max_length], dtype=tf.int32) * self.vocab.end_token_id
            # [batch_size, max_time_steps + 1]
            self.targets_sparse = tf.concat([targets, end_token], axis=1)
            self.targets_length = targets_length + 1
            self.target_mask = tf.sequence_mask(
                lengths=self.targets_length, maxlen=self.hparams.max_length, dtype=tf.float32)

    def build_cell(self):
        sequence_length = tf.ones(
            [self.batch_size], dtype=tf.int32) * self.hparams.max_length
        super().build_cell(sequence_length, self.src_vocab.size())

    def build_obj(self):
        output_layer = Dense(self.tgt_vocab.size(), name='output_projection')
        self.logits = output_layer(self.outputs)

    def build_rt_decoder(self):
        with tf.name_scope('predict_decoder'):
            self.hypotheses = tf.argmax(self.logits, -1)
            self.scores = tf.reduce_sum(tf.reduce_max(
                tf.nn.log_softmax(self.logits), -1), -1)
