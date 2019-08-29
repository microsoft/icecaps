import tensorflow as tf
import numpy as np
import math

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator
from icecaps.util.vocabulary import Vocabulary
import icecaps.util.trees as trees


class AbstractTransformerEstimator(AbstractIcecapsEstimator):

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["vocab_file"] = cls.make_param(
            "icecaps/examples/dummy_data/vocab.dic")
        expected_params["vocab_size"] = cls.make_param(0)
        expected_params["depth"] = cls.make_param(1)
        expected_params["num_heads"] = cls.make_param(8)
        expected_params["d_model"] = cls.make_param(32)
        expected_params["d_pos"] = cls.make_param(32)
        expected_params["d_ff"] = cls.make_param(64)
        expected_params["max_length"] = cls.make_param(10)
        expected_params["min_wavelength"] = cls.make_param(1.0)
        expected_params["max_wavelength"] = cls.make_param(1000.0)
        expected_params["warmup_steps"] = cls.make_param(4000.0)
        expected_params["fixed_learning_rate"] = cls.make_param(False)
        expected_params["learn_wavelengths"] = cls.make_param(False)
        expected_params["modality"] = cls.make_param("seq")
        expected_params["tree_depth"] = cls.make_param(256)
        expected_params["tree_width"] = cls.make_param(2)
        expected_params["learn_positional_embeddings"] = cls.make_param(False)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        self.d_k = self.hparams.d_model // self.hparams.num_heads
        self.d_pos = self.hparams.d_pos if self.hparams.d_pos == 0 else self.hparams.d_pos
        self.d_ff = self.hparams.d_ff if self.hparams.d_ff == 0 else self.hparams.d_ff
        if self.hparams.vocab_size > 0:
            self.vocab = Vocabulary(size=self.hparams.vocab_size)
        else:
            self.vocab = Vocabulary(fname=self.hparams.vocab_file)
        if not self.hparams.fixed_learning_rate:
            self.train_step = tf.get_variable(
                'train_step', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)
            self.learning_rate = (  # magic formula provided in transformer paper
                tf.sqrt(1.0 / self.hparams.d_model) * tf.minimum(self.train_step * tf.pow(self.hparams.warmup_steps, -1.5), tf.pow(self.train_step, -0.5)))

    def build_embeddings(self):
        # self.hparams.max_length), dtype=tf.float32), 1)
        position = tf.expand_dims(
            tf.cast(tf.range(0, 2048), dtype=tf.float32), 1)
        if self.hparams.learn_wavelengths:
            wavelength_logs = tf.get_variable(
                "wavelength_logs", [self.d_pos // 2], tf.float32)
        else:
            wavelength_logs = tf.linspace(math.log(self.hparams.min_wavelength), math.log(
                self.hparams.max_wavelength), self.d_pos // 2)
        div_term = tf.expand_dims(tf.exp(-wavelength_logs), 0)
        outer_product = tf.matmul(position, div_term)
        cosines = tf.cos(outer_product)
        sines = tf.sin(outer_product)
        self.positional_embeddings = tf.concat([cosines, sines], -1)
        if self.hparams.learn_positional_embeddings:
            self.positional_embeddings = tf.get_variable(
                name='positional_embeddings', shape=[self.hparams.max_length, self.hparams.d_model]) * np.sqrt(float(self.hparams.d_model))            
        self.token_embeddings = tf.get_variable(
            name='token_embeddings', shape=[self.vocab.size(), self.hparams.d_model]) * np.sqrt(float(self.hparams.d_model))
        if self.hparams.modality == "tree":
            self.d_tree_param = self.d_pos // (
                self.hparams.tree_depth * self.hparams.tree_width)
            self.tree_params = tf.tanh(tf.get_variable(
                "tree_params", [self.d_tree_param]))
            self.tiled_tree_params = tf.tile(tf.reshape(self.tree_params, [
                                             1, 1, -1]), [self.hparams.tree_depth, self.hparams.tree_width, 1])
            self.tiled_depths = tf.tile(tf.reshape(tf.range(self.hparams.tree_depth, dtype=tf.float32), [
                                        -1, 1, 1]), [1, self.hparams.tree_width, self.d_tree_param])
            self.tree_norm = tf.sqrt(
                (1 - tf.square(self.tree_params)) * self.hparams.d_model / 2)
            self.tree_weights = tf.reshape(tf.pow(self.tiled_tree_params, self.tiled_depths) * self.tree_norm,
                                           [self.hparams.tree_depth * self.hparams.tree_width, self.d_tree_param])

    def treeify_positions(self, positions):
        treeified = tf.expand_dims(positions, -1) * self.tree_weights
        shape = tf.shape(treeified)
        shape = tf.concat([shape[:-2], [self.d_pos]], -1)
        treeified = tf.reshape(treeified, shape)
        return treeified

    def init_inputs(self):
        self.inputs_sparse = tf.cast(self.features["inputs"], tf.int32)
        self.mask = tf.cast(tf.not_equal(
            self.inputs_sparse, self.vocab.end_token_id), tf.float32)
        self.inputs_length = tf.cast(tf.count_nonzero(self.mask, -1), tf.int32)
        self.inputs_max_length = tf.reduce_max(self.inputs_length)
        self.batch_size = tf.shape(self.inputs_sparse)[0]
        self.inputs_sparse = tf.slice(self.inputs_sparse, [0, 0], [
                               self.batch_size, self.inputs_max_length])
        self.mask = tf.slice(self.mask, [0, 0], [
                             self.batch_size, self.inputs_max_length])
        self.inputs_dense = tf.nn.embedding_lookup(
            params=self.token_embeddings, ids=self.inputs_sparse)
        if self.hparams.modality == "seq":
            self.positions = tf.slice(self.positional_embeddings, [0, 0], [
                                      self.inputs_max_length, self.d_pos])
        elif self.hparams.modality == "tree":
            self.positions = tf.reshape(self.features["inputs_positions"], [
                                        self.batch_size, self.inputs_max_length,
                                        self.hparams.tree_depth * self.hparams.tree_width])
            self.positions = self.treeify_positions(self.positions)
        else:
            raise ValueError("This input modality is not supported.")
        if self.d_pos != self.hparams.d_model:
            self.positions = tf.layers.dense(
                self.positions, self.hparams.d_model)
        self.inputs_dense = self.inputs_dense + self.positions
        self.inputs_dense = tf.nn.dropout(
            self.inputs_dense, self.keep_prob)
        self.inputs_dense = tf.transpose(tf.transpose(
            self.inputs_dense) * tf.transpose(self.mask))

    def build_layer_norm(self, x):
        return tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)

    def build_sublayer_fn(self, x, f):
        x = self.build_layer_norm(x)
        x = x + tf.nn.dropout(f(x), self.keep_prob)
        return x

    def attention(self, query, key, value, d_k, enc_mask=None, dec_mask=None):
        scores = tf.matmul(query, tf.transpose(
            key, [0, 1, 3, 2])) / math.sqrt(d_k)
        if enc_mask is not None:
            scores = tf.transpose(
                scores, [1, 2, 0, 3]) * enc_mask - 1e24 * (1.0 - enc_mask)
            scores = tf.transpose(scores, [2, 0, 1, 3])
        if dec_mask is not None:
            scores = scores * dec_mask - 1e24 * (1.0 - dec_mask)
        p_attn = tf.nn.softmax(scores)
        p_attn = tf.nn.dropout(p_attn, keep_prob=self.keep_prob)
        attended_values = tf.matmul(p_attn, value)
        return attended_values, p_attn

    def mha_fn(self, query, key, value, batch_size, enc_mask_, dec_mask_):
        with tf.variable_scope("mha", reuse=tf.AUTO_REUSE) as scope:
            query = tf.transpose(tf.reshape(tf.layers.dense(query, self.hparams.d_model, use_bias=True), [
                                 batch_size, -1, self.hparams.num_heads, self.d_k]), [0, 2, 1, 3])
            key = tf.transpose(tf.reshape(tf.layers.dense(key, self.hparams.d_model, use_bias=True), [
                               batch_size, -1, self.hparams.num_heads, self.d_k]), [0, 2, 1, 3])
            value = tf.transpose(tf.reshape(tf.layers.dense(value, self.hparams.d_model, use_bias=True), [
                                 batch_size, -1, self.hparams.num_heads, self.d_k]), [0, 2, 1, 3])
            attended, _ = self.attention(query, key, value, self.d_k, enc_mask_, dec_mask_)
            attended = tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), [
                                  batch_size, -1, self.hparams.d_model])
            return attended

    def build_mha_sublayer(self, x, m, batch_size, enc_mask=None, dec_mask=None):
        with tf.variable_scope("attn", reuse=tf.AUTO_REUSE) as scope:
            return self.build_sublayer_fn(
                x, lambda q: tf.layers.dense(self.mha_fn(q, m, m, batch_size, enc_mask, dec_mask), self.hparams.d_model))

    def build_ffn_sublayer(self, x, d_ff):
        with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE) as scope:
            def ffn_fn(q): return tf.layers.dense(
                tf.layers.dense(q, d_ff, tf.nn.relu), self.hparams.d_model)
            return self.build_sublayer_fn(x, ffn_fn)

    def build_optimizer(self, trainable_params=None):
        super().build_optimizer(trainable_params)
        self.step_update_op = tf.assign_add(self.train_step, 1.0)
        with tf.control_dependencies([self.step_update_op]):
            self.train_op = tf.group([self.step_update_op, self.train_op])
