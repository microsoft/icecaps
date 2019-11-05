import os
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, ResidualWrapper
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator
from icecaps.util.vocabulary import Vocabulary


class AbstractRecurrentEstimator(AbstractIcecapsEstimator):

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["max_length"] = cls.make_param(50)
        expected_params["cell_type"] = cls.make_param('gru')
        expected_params["hidden_units"] = cls.make_param(32)
        expected_params["depth"] = cls.make_param(1)
        expected_params["token_embed_dim"] = cls.make_param(16)
        expected_params["tie_token_embeddings"] = cls.make_param(True)
        expected_params["beam_width"] = cls.make_param(8)
        expected_params["vocab_file"] = cls.make_param("./dummy_data/vocab.dic")
        expected_params["vocab_size"] = cls.make_param(0)
        expected_params["skip_tokens"] = cls.make_param('')
        expected_params["skip_tokens_start"] = cls.make_param('')
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        if self.hparams.vocab_size > 0:
            self.vocab = Vocabulary(size=self.hparams.vocab_size)
        else:
            self.vocab = Vocabulary(fname=self.hparams.vocab_file, skip_tokens=self.hparams.skip_tokens, skip_tokens_start=self.hparams.skip_tokens_start)

    def build_cell(self, name=None):
        if self.hparams.cell_type == 'linear':
            cell = BasicRNNCell(self.hparams.hidden_units,
                                activation=tf.identity, name=name)
        elif self.hparams.cell_type == 'tanh':
            cell = BasicRNNCell(self.hparams.hidden_units,
                                activation=tf.tanh, name=name)
        elif self.hparams.cell_type == 'relu':
            cell = BasicRNNCell(self.hparams.hidden_units,
                                activation=tf.nn.relu, name=name)
        elif self.hparams.cell_type == 'gru':
            cell = GRUCell(self.hparams.hidden_units, name=name)
        elif self.hparams.cell_type == 'lstm':
            cell = LSTMCell(self.hparams.hidden_units, name=name, state_is_tuple=False)
        else:
            raise ValueError('Provided cell type not supported.')
        return cell

    def build_deep_cell(self, cell_list=None, name=None, return_raw_list=False):
        if name is None:
            name = "cell"
        if cell_list is None:
            cell_list = []
            for i in range(self.hparams.depth):
                cell = self.build_cell(name=name+"_"+str(i))
                cell = DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cell_list.append(cell)
        if return_raw_list:
            return cell_list
        if len(cell_list) == 1:
            return cell_list[0]
        return MultiRNNCell(cell_list, state_is_tuple=False)

    def build_rnn(self, input_key="inputs"):
        with tf.variable_scope('rnn'):
            self.cell = self.build_deep_cell()
            self.build_inputs(input_key)
            self.outputs, self.last_state = tf.nn.dynamic_rnn(
                cell=self.cell, inputs=self.inputs_dense,
                sequence_length=self.inputs_length,
                time_major=False, dtype=tf.float32)  # [batch_size, max_time_step, cell_output_size], [batch_size, cell_output_size]

    def build_embeddings(self):
        if "token_embeddings" in self.features and self.hparams.tie_token_embeddings:
            self.token_embeddings = self.features["token_embeddings"]
        else:
            self.token_embeddings = tf.get_variable(
                name='token_embeddings', shape=[self.vocab.size(), self.hparams.token_embed_dim])
            if self.hparams.token_embed_dim != self.hparams.hidden_units:
                projection = tf.get_variable(
                    name='token_embed_proj', shape=[self.hparams.token_embed_dim, self.hparams.hidden_units])
                self.token_embeddings = self.token_embeddings @ projection

    def embed_sparse_to_dense(self, sparse):
        with tf.variable_scope('embed_sparse_to_dense', reuse=tf.AUTO_REUSE):
            dense = tf.nn.embedding_lookup(self.token_embeddings, sparse)
        return dense

    def build_inputs(self, input_key):
        self.build_embeddings()
        self.inputs_sparse_untrimmed = tf.cast(self.features[input_key], tf.int32)
        self.inputs_length = tf.cast(tf.count_nonzero(
            self.inputs_sparse_untrimmed - self.vocab.end_token_id, -1), tf.int32)
        self.inputs_max_length = tf.reduce_max(self.inputs_length)
        self.inputs_sparse = tf.slice(self.inputs_sparse_untrimmed, [0, 0], [-1, self.inputs_max_length])
        self.inputs_dense = self.embed_sparse_to_dense(self.inputs_sparse)
        self.batch_size = tf.shape(self.inputs_sparse)[0]

    def build_loss(self):
        with tf.name_scope('build_loss'):
            self.loss = seq2seq.sequence_loss(
                logits=self.logits, targets=self.targets_sparse, weights=self.target_mask,
                average_across_timesteps=True, average_across_batch=True,)
        self.reported_loss = tf.identity(self.loss, 'reported_loss')
