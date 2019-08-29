import tensorflow as tf
import copy
import random
from collections import OrderedDict

from tensorflow.python.layers.core import Dense
from tensorflow.nn.rnn_cell import LSTMStateTuple

from icecaps.estimators.estimator_group import EstimatorGroup
from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.seq2seq_encoder_estimator import Seq2SeqEncoderEstimator
from icecaps.estimators.seq2seq_decoder_estimator import Seq2SeqDecoderEstimator
from icecaps.estimators.noise_layer import NoiseLayer
from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator
from icecaps.estimators.abstract_recurrent_estimator import AbstractRecurrentEstimator
from icecaps.estimators.abstract_transformer_estimator import AbstractTransformerEstimator
from icecaps.estimators.convolutional_estimator import ConvolutionalEstimator
from icecaps.util.vocabulary import Vocabulary


class CmrEncoderEstimator(AbstractRecurrentEstimator):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope=""):
        super().__init__(model_dir, params, config, scope)

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.init_inputs()
            self.build_lexical_embeddings()
            self.query_outputs, _ = self.build_contextual_encoding(
                self.query, "query")
            self.document_outputs, _ = self.build_contextual_encoding(
                self.document, "document")
            self.outputs, self.last_state = self.build_memory()
            if mode == tf.estimator.ModeKeys.PREDICT:
                self.predictions = self.flatten_nested_tensors({
                    "inputs": self.features["inputs"],
                    "document": self.document,
                    "outputs": self.outputs,
                    "state": self.last_state,
                    "length": self.query_length,
                    "token_embeddings": self.token_embeddings,
                })
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_optimizer()
                raise NotImplementedError(
                    "Training not currently supported for seq2seq encoder.")
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = dict()
                raise NotImplementedError(
                    "Evaluation not currently supported for seq2seq encoder.")

    def init_inputs(self):
        self.query = self.features["inputs"]
        self.document = self.features["document"]
        self.batch_size = tf.shape(self.query)[0]
        self.query_mask = tf.cast(tf.not_equal(
            self.query, self.vocab.end_token_id), tf.float32)
        self.query_length = tf.cast(tf.count_nonzero(
            self.query - self.vocab.end_token_id, -1), tf.int32)

    def build_lexical_embeddings(self):
        self.token_embeddings = tf.get_variable(
            name='embedding', shape=[self.vocab.size(), self.hparams.token_embed_dim])
        if "embeddings" in self.features:
            self.token_embeddings = lexical_dnn.model_fn(
                self.features["embeddings"], tf.estimator.ModeKeys.PREDICT, self.params)

    def build_cell(self, signal, length, name=''):
        cell_fw = self.build_multi_cell(name=name+'/qfw')
        cell_bw = self.build_multi_cell(name=name+'/qbw')
        contextual_outputs, contextual_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=signal, sequence_length=length,
            time_major=False, dtype=tf.float32)
        contextual_outputs = contextual_outputs[0] + contextual_outputs[1]
        contextual_state = contextual_state[0]
        return contextual_outputs, contextual_state

    def build_contextual_encoding(self, inputs_, name=""):
        embedded = tf.nn.embedding_lookup(
            params=self.token_embeddings, ids=inputs_)
        if self.hparams.use_embedding_projection:
            projection = Dense(self.hparams.hidden_units,
                               name=name+'_input_projection')
            embedded = projection(embedded)
        length = tf.cast(tf.count_nonzero(
            self.query - self.vocab.end_token_id, -1), tf.int32)
        return self.build_cell(embedded, length, name + "/contextual")

    def attention(self, query, key, value, mask):
        scores = tf.matmul(query, tf.transpose(
            key, [0, 2, 1])) / tf.sqrt(float(self.hparams.hidden_units))
        scores = tf.transpose(scores, [1, 0, 2]) * mask - 1e24 * (1.0 - mask)
        scores = tf.transpose(scores, [1, 0, 2])
        p_attn = tf.nn.softmax(scores)
        p_attn = tf.nn.dropout(p_attn, keep_prob=self.keep_prob)
        attended_values = tf.matmul(p_attn, value)
        return attended_values

    def build_memory(self):
        query_keys = tf.layers.dense(
            self.query_outputs, self.hparams.hidden_units)
        query_values = tf.layers.dense(
            self.query_outputs, self.hparams.hidden_units)
        signal = self.attention(
            self.document_outputs, query_keys, query_values, self.query_mask)
        signal_keys = tf.layers.dense(signal, self.hparams.hidden_units)
        signal_values = tf.layers.dense(signal, self.hparams.hidden_units)
        signal = self.attention(signal, signal_keys,
                                signal_values, self.query_mask)
        return self.build_cell(signal, self.query_length, "memory")

    def build_loss(self):
        raise NotImplementedError("No loss function for CMR encoder.")
