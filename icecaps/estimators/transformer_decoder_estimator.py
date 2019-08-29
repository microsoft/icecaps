import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq

from icecaps.estimators.abstract_transformer_estimator import AbstractTransformerEstimator
import icecaps.util.trees as trees


class TransformerDecoderEstimator(AbstractTransformerEstimator):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope="", is_mmi_model=False):
        self.is_mmi_model = is_mmi_model
        super().__init__(model_dir, params, config, scope)

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
            self.extract_args(features, mode, params)
            self.build_embeddings()
            self.encoder_outputs = self.features["inputs"]
            self.encoder_mask = self.features["mask"]
            self.batch_size = tf.shape(self.encoder_outputs)[0]
            self.build_io_layers()
            if mode == tf.estimator.ModeKeys.PREDICT:
                if self.is_mmi_model:
                    self.init_targets()
                    self.build_train_decoder()
                    self.build_loss(False)
                    self.predictions = {
                        "inputs": tf.convert_to_tensor(self.features["original_inputs"]),
                        "targets": tf.convert_to_tensor(self.features["targets"]),
                        "scores": self.loss,
                    }
                else:
                    self.build_rt_decoder()
                    self.predictions = {
                        "inputs": self.inputs_pred,
                        "scores": self.predicted_scores,
                        "outputs": self.predicted_hypotheses,
                        "positions": self.predicted_positions,
                    }
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            self.init_targets()
            self.build_train_decoder()
            self.build_loss()
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_optimizer()
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, train_op=self.train_op)
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.build_rt_decoder()
                self.predicted_hypotheses = tf.reshape(
                    self.predicted_hypotheses, [self.batch_size, self.hparams.beam_width, -1])[:, 0, :self.targets_max_length]
                self.eval_metric_ops = {
                    "token_accuracy_gs": tf.metrics.accuracy(
                        labels=self.targets_train, predictions=self.gs_hypotheses, weights=self.target_mask),
                    "token_accuracy_true": tf.metrics.accuracy(
                        labels=self.targets_train, predictions=self.predicted_hypotheses, weights=self.target_mask),
                    "whole_sequence_accuracy_gs": tf.metrics.accuracy(
                        labels=tf.zeros([self.batch_size], dtype=tf.int32),
                        predictions=tf.reduce_sum(tf.cast(self.target_mask, tf.int32) * tf.squared_difference(
                            self.targets_train, tf.cast(self.gs_hypotheses, tf.int32)), -1)),
                    "whole_sequence_accuracy_true": tf.metrics.accuracy(
                        labels=tf.zeros([self.batch_size], dtype=tf.int32),
                        predictions=tf.reduce_sum(tf.cast(self.target_mask, tf.int32) * tf.squared_difference(
                            self.targets_train, tf.cast(self.predicted_hypotheses, tf.int32)), -1)),
                }
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, eval_metric_ops=self.eval_metric_ops)

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["tie_token_embeddings"] = cls.make_param(True)
        expected_params["traversal"] = cls.make_param("dfs")
        expected_params["beam_width"] = cls.make_param(5)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        if self.hparams.modality == "tree":
            self.degrees = tf.cast(trees.get_degrees(
                self.vocab.idx2word), tf.int32)

    def build_embeddings(self):
        super().build_embeddings()
        if self.hparams.tie_token_embeddings:
            self.token_embeddings = self.features["token_embeddings"]
        else:
            self.token_embeddings = tf.get_variable(
                name='token_embeddings', 
                shape=[self.vocab.size(), self.hparams.d_model]) * np.sqrt(float(self.hparams.d_model))

    def init_targets(self):
        inputs_sparse = tf.cast(self.features["targets"], tf.int32)
        inputs_length = tf.cast(tf.count_nonzero(
            inputs_sparse - self.vocab.end_token_id, -1), tf.int32)
        inputs_max_length = tf.reduce_max(inputs_length)
        inputs_sparse = tf.slice(inputs_sparse, [0, 0], [-1, inputs_max_length])
        start_token = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * self.vocab.start_token_id
        end_token = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * self.vocab.end_token_id
        # [batch_size, max_time_steps + 1]
        self.inputs_train = tf.concat([start_token, inputs_sparse], axis=-1)
        if self.hparams.modality == "seq":
            # [batch_size, max_time_steps + 1]
            self.targets_train = tf.concat([inputs_sparse, end_token], axis=-1)
            self.targets_max_length = inputs_max_length + 1
            self.positions = tf.slice(self.positional_embeddings, [0, 0], [
                                      self.targets_max_length, self.d_pos])
        elif self.hparams.modality == "tree":
            self.inputs_train = self.inputs_train[:, :-1]
            self.targets_train = inputs_sparse
            self.targets_max_length = inputs_max_length
            self.positions = tf.reshape(self.features["targets_positions"], [
                                        self.batch_size, self.targets_max_length, self.hparams.tree_depth * self.hparams.tree_width])
            self.positions = self.treeify_positions(self.positions)
            self.positions = tf.concat(
                [tf.zeros([self.batch_size, 1, self.d_pos]), self.positions[:, :-1, :]], -2)
        else:
            raise ValueError("This output modality is not supported.")
        self.mask_length = self.targets_max_length
        if self.d_pos != self.hparams.d_model:
            self.positions = self.pos_proj_layer(self.positions)
        self.target_mask = tf.cast(tf.not_equal(
            self.inputs_train, self.vocab.end_token_id), tf.float32)
        self.inputs_dense = tf.nn.embedding_lookup(
            params=self.token_embeddings, ids=self.inputs_train)
        self.inputs_dense = self.inputs_dense + self.positions
        self.inputs_dense = tf.nn.dropout(
            self.inputs_dense, self.keep_prob)
        self.inputs_dense = tf.transpose(tf.transpose(
            self.inputs_dense) * tf.transpose(self.target_mask))

    def build_layer(self, x, m, batch_size, mask):
        signal = self.build_mha_sublayer(
            x, x, batch_size, enc_mask=None, dec_mask=mask)
        signal = self.build_mha_sublayer(
            signal, m, batch_size, enc_mask=self.encoder_mask, dec_mask=None)
        signal = self.build_ffn_sublayer(signal, self.d_ff)
        return signal

    def core_fn(self, inputs_sparse, encoder_outputs, batch_size, mask):
        with tf.variable_scope("core", reuse=tf.AUTO_REUSE) as scope:
            signal = inputs_sparse  # self.build_layer_norm(inputs_sparse)
            for i in range(self.hparams.depth):
                with tf.variable_scope("layer_" + str(i), reuse=tf.AUTO_REUSE) as scope:
                    signal = self.build_layer(
                        signal, encoder_outputs, batch_size, mask)
            return signal

    def build_layer_rt(self, next_x, prev_x, m, batch_size, tiled_encoder_mask):
        signal = self.build_mha_sublayer(next_x, tf.concat(
            [prev_x, next_x], -2), batch_size, enc_mask=None, dec_mask=None)
        signal = self.build_mha_sublayer(
            signal, m, batch_size, enc_mask=tiled_encoder_mask, dec_mask=None)
        signal = self.build_ffn_sublayer(signal, self.d_ff)
        return signal

    def core_fn_rt(self, next_x, prev_values, encoder_outputs, batch_size, tiled_encoder_mask):
        with tf.variable_scope("core", reuse=tf.AUTO_REUSE) as scope:
            signals = []
            signals.append(next_x) #self.build_layer_norm(next_x))
            for i in range(self.hparams.depth):
                with tf.variable_scope("layer_" + str(i), reuse=tf.AUTO_REUSE) as scope:
                    if prev_values is not None:
                        signals.append(self.build_layer_rt(
                            signals[i], prev_values[i], encoder_outputs, batch_size, tiled_encoder_mask))
                    else:
                        signals.append(self.build_layer(
                            signals[i], encoder_outputs, batch_size, None))
            return signals

    def build_io_layers(self):
        if self.hparams.tie_token_embeddings:
            self.output_embeddings = self.features["token_embeddings"] / \
                np.sqrt(float(self.hparams.d_model))
            self.output_layer = lambda x: tf.reshape(
                tf.matmul(tf.reshape(x, [-1, self.hparams.d_model]), self.output_embeddings, transpose_b=True), [self.batch_size, -1, self.vocab.size()])
        else:
            self.output_layer = tf.layers.Dense(self.vocab.size(), name="out_layer")
        if self.d_pos != self.hparams.d_model:
            self.pos_proj_layer = tf.layers.Dense(self.hparams.d_model, name="pos_proj")

    def get_mask(self, dim):
        return tf.matrix_band_part(tf.ones([dim, dim]), -1, 0)

    def build_train_decoder(self):
        mask = self.get_mask(self.mask_length)
        outputs = self.core_fn(self.inputs_dense, self.encoder_outputs, self.batch_size, mask)
        self.outputs = outputs
        self.logits = self.output_layer(outputs)
        self.logits = tf.reshape(
            self.logits, [self.batch_size, self.targets_max_length, self.vocab.size()])
        self.probs = tf.nn.log_softmax(self.logits)
        self.gs_hypotheses = tf.argmax(self.probs, -1)

    def build_loss(self, average_across_batch=True):
        with tf.name_scope('build_loss'):
            self.loss = seq2seq.sequence_loss(
                logits=self.logits, targets=self.targets_train, weights=self.target_mask,
                average_across_timesteps=True, average_across_batch=average_across_batch,)
        self.reported_loss = tf.identity(self.loss, 'reported_loss')

    def get_next_pos_seq(self, t, effective_batch_size):
        return tf.tile(tf.reshape(self.positional_embeddings[t], [1, 1, -1]), [effective_batch_size, 1, 1])

    def is_complete_seq(self, hypotheses, shape):
        if hypotheses is None:
            return None
        return tf.equal(hypotheses[:, -1], tf.ones(shape=shape, dtype=tf.int32) * self.vocab.end_token_id)

    def get_next_pos_tree(self, hypotheses, positions, aux, effective_batch_size):
        placeholder_pos = tf.zeros(
            [effective_batch_size, 1, self.hparams.tree_width * self.hparams.tree_depth])
        placeholder_aux = tf.zeros([effective_batch_size, 1], dtype=tf.int32)
        if positions is None:
            next_pos = placeholder_pos
        else:
            if self.hparams.traversal == "dfs":
                # find the highest-index position with nonzero children
                expand_at = (tf.shape(
                    aux)[-1] - 1) - tf.argmax(tf.reverse(aux, [-1]), -1, output_type=tf.int32)
            elif self.hparams.traversal == "bfs":
                # find the lowest-index position with nonzero children
                expand_at = tf.argmax(tf.sign(aux), -1, output_type=tf.int32)
            expand_at_gnd = tf.concat([
                tf.expand_dims(tf.range(effective_batch_size), -1),
                tf.reshape(expand_at, [-1, 1])], -1)
            branches = tf.one_hot(
                self.hparams.tree_width - tf.gather_nd(aux, expand_at_gnd), self.hparams.tree_width)
            next_pos = tf.gather_nd(positions, expand_at_gnd)
            next_pos = tf.concat(
                [branches, next_pos], -1)[:, : self.hparams.tree_width * self.hparams.tree_depth]
            next_pos = tf.expand_dims(next_pos, 1)
            complete_indices = self.is_complete_tree(
                aux, [effective_batch_size])
            next_pos = tf.where(complete_indices, placeholder_pos, next_pos)
            aux -= tf.one_hot(expand_at, tf.shape(aux)[-1], dtype=tf.int32)
        degree_query = hypotheses[:, -1:]
        next_aux = tf.gather_nd(self.degrees, degree_query)
        next_aux = tf.reshape(next_aux, [effective_batch_size, 1])
        if positions is not None:
            next_aux = tf.where(complete_indices, placeholder_aux, next_aux)
        if aux is not None:
            aux = tf.concat([aux, next_aux], -1)
        else:
            aux = next_aux
        return next_pos, aux

    def is_complete_tree(self, aux, shape):
        if aux is None:
            return None
        return tf.equal(tf.reduce_sum(tf.nn.relu(aux), -1), tf.zeros(shape=shape, dtype=tf.int32))

    def get_dense_start_tokens(self, effective_batch_size):
        start_tokens = tf.ones(
            shape=[effective_batch_size, 1], dtype=tf.int32) * self.vocab.start_token_id
        return tf.nn.embedding_lookup(
            params=self.token_embeddings, ids=start_tokens)

    def single_best(self, hypotheses, curr_beam_width, t):
        hypotheses = hypotheses[:, t:t+1]
        dummy = tf.ones([self.batch_size, curr_beam_width - 1], dtype=tf.int32) * self.vocab.end_token_id
        padded = tf.concat([hypotheses, dummy], -1)
        scores = tf.concat([tf.zeros([self.batch_size, 1]), -10000 * tf.ones([self.batch_size, curr_beam_width - 1])], -1)
        return scores, padded

    def build_rt_decoder(self):
        scores = None
        hypotheses = None
        prev_signals = None
        positions = None
        aux = None
        curr_beam_width = 1
        effective_batch_size = self.batch_size
        tiled_encoder_outputs = self.encoder_outputs
        tiled_encoder_mask = self.encoder_mask

        for t in range(self.hparams.max_length):
            # create positions and inputs_sparse
            if self.hparams.modality == "seq":
                start_positions = self.get_next_pos_seq(
                    0, effective_batch_size)
            elif self.hparams.modality == "tree":
                start_positions = tf.zeros(
                    [effective_batch_size, 1, self.hparams.tree_width * self.hparams.tree_depth])
            if positions is not None:
                concat_positions = tf.concat([start_positions, positions], -2)
            else:
                concat_positions = start_positions
            if self.hparams.modality == "tree":
                concat_positions = self.treeify_positions(concat_positions)
            concat_positions = tf.reshape(
                concat_positions, [-1, t+1, self.d_pos])
            if self.d_pos != self.hparams.d_model:
                concat_positions = self.pos_proj_layer(concat_positions)
            start_tokens_dense = self.get_dense_start_tokens(effective_batch_size)
            if hypotheses is None:
                inputs_dense = start_tokens_dense
            else:
                hypotheses_dense = tf.nn.embedding_lookup(
                    params=self.token_embeddings, ids=hypotheses)
                inputs_dense = tf.concat([start_tokens_dense, hypotheses_dense], -2)
            inputs_dense = tf.reshape(
                inputs_dense, [effective_batch_size, -1, self.hparams.d_model])
            inputs_dense = inputs_dense + concat_positions

            # calculate all possible scores for next states for current states
            next_signals = self.core_fn_rt(
                inputs_dense[:, -1:, :], prev_signals, tiled_encoder_outputs, effective_batch_size, tiled_encoder_mask)
            logits = self.output_layer(next_signals[-1])
            logits = tf.nn.log_softmax(logits)
            if hypotheses is not None:
                if self.hparams.modality == "seq":
                    complete_indices = self.is_complete_seq(
                        hypotheses, [effective_batch_size])
                elif self.hparams.modality == "tree":
                    complete_indices = self.is_complete_tree(
                        aux, [effective_batch_size])
            else:
                complete_indices = None
            if complete_indices is not None:
                complete_indices = tf.reshape(
                    complete_indices, [effective_batch_size])
            else:
                complete_indices = tf.equal(
                    tf.zeros(shape=[effective_batch_size]), tf.ones(shape=[effective_batch_size]))
            complete_logits = tf.ones(
                [effective_batch_size], dtype=tf.int32) * self.vocab.end_token_id
            complete_logits = tf.log(tf.one_hot(
                complete_logits, self.vocab.size()) + 1e-12)
            logits = tf.reshape(
                logits, [effective_batch_size, self.vocab.size()])
            logits = tf.where(complete_indices, complete_logits, logits)

            # trim beams
            if t + 1 < self.hparams.max_length:
                logits = tf.reshape(
                    logits, [self.batch_size, curr_beam_width, self.vocab.size()])
                logits = tf.transpose(logits, [2, 0, 1])
                scores = (logits + scores) if scores is not None else logits
                scores = tf.transpose(scores, [1, 2, 0])
                scores = tf.reshape(
                    scores, [self.batch_size, curr_beam_width*self.vocab.size()])

                # update beam width
                if curr_beam_width < self.hparams.beam_width:
                    if t == self.hparams.max_length:
                        curr_beam_width = 1
                    else:
                        curr_beam_width *= self.vocab.size()
                        if curr_beam_width > self.hparams.beam_width:
                            curr_beam_width = self.hparams.beam_width
                    effective_batch_size = self.batch_size * curr_beam_width
                    if self.encoder_outputs is not None:
                        tiled_encoder_outputs = tf.tile(tf.expand_dims(
                            self.encoder_outputs, 1), [1, curr_beam_width, 1, 1])
                        tiled_encoder_outputs = tf.reshape(
                            tiled_encoder_outputs, [effective_batch_size, -1, self.hparams.d_model])
                        tiled_encoder_mask = tf.tile(tf.expand_dims(
                            self.encoder_mask, 1), [1, curr_beam_width, 1])
                        tiled_encoder_mask = tf.reshape(
                            tiled_encoder_mask, [effective_batch_size, -1])

                # choose top k scores
                scores, beam_indices = tf.nn.top_k(scores, curr_beam_width)
                beam_gather_indices = tf.concat([
                    tf.expand_dims(
                        tf.range(effective_batch_size) // curr_beam_width, 1),
                    tf.reshape(beam_indices // self.vocab.size(), [-1, 1])], 1)

                if t == 0:
                    hypotheses = tf.reshape(
                        beam_indices, [effective_batch_size, 1])
                    prev_signals = []
                    for k in range(self.hparams.depth):
                        prev_signals.append(tf.reshape(tf.gather_nd(next_signals[k], beam_gather_indices), [
                                            effective_batch_size, t+1, self.hparams.d_model]))
                else:
                    hypotheses = tf.reshape(
                        hypotheses, [self.batch_size, curr_beam_width, t])
                    hypotheses = tf.gather_nd(hypotheses, beam_gather_indices)
                    hypotheses = tf.concat([
                        tf.reshape(hypotheses, [effective_batch_size, t]),
                        tf.reshape(tf.mod(beam_indices, self.vocab.size()), [effective_batch_size, 1])], 1)
                    for k in range(self.hparams.depth):
                        prev_signals[k] = tf.concat(
                            [prev_signals[k], next_signals[k]], -2)
                        prev_signals[k] = tf.reshape(
                            prev_signals[k], [self.batch_size, curr_beam_width, t+1, self.hparams.d_model])
                        prev_signals[k] = tf.reshape(tf.gather_nd(prev_signals[k], beam_gather_indices), [
                                                     effective_batch_size, t+1, self.hparams.d_model])
                    if self.hparams.modality == "seq":
                        positions = tf.reshape(
                            positions, [self.batch_size, curr_beam_width, t, self.d_pos])
                        positions = tf.reshape(tf.gather_nd(positions, beam_gather_indices), [
                                               effective_batch_size, t, self.d_pos])
                    elif self.hparams.modality == "tree":
                        positions = tf.reshape(positions, [
                                               self.batch_size, curr_beam_width, t, self.hparams.tree_width * self.hparams.tree_depth])
                        positions = tf.reshape(tf.gather_nd(positions, beam_gather_indices), [
                                               effective_batch_size, t, self.hparams.tree_width * self.hparams.tree_depth])
                    if aux is not None:
                        aux = tf.reshape(
                            aux, [self.batch_size, curr_beam_width, t])
                        aux = tf.reshape(tf.gather_nd(aux, beam_gather_indices), [
                                         effective_batch_size, t])

                # compute next position
                if self.hparams.modality == "seq":
                    next_pos = self.get_next_pos_seq(t+1, effective_batch_size)
                elif self.hparams.modality == "tree":
                    next_pos, aux = self.get_next_pos_tree(
                        hypotheses, positions, aux, effective_batch_size)
                else:
                    raise ValueError(
                        "Beam search decoding is not available for this modality.")
                if positions is not None:
                    positions = tf.concat([positions, next_pos], -2)
                else:
                    positions = next_pos

        self.inputs_pred = tf.reshape(tf.tile(tf.expand_dims(self.features["original_inputs"], 1), [
                                      1, curr_beam_width, 1]), [self.batch_size*curr_beam_width, -1])
        self.predicted_scores = tf.reshape(scores, [self.batch_size*curr_beam_width])
        self.predicted_hypotheses = tf.reshape(hypotheses, [self.batch_size*curr_beam_width, t])
        self.predicted_positions = tf.reshape(
            positions, [self.batch_size*curr_beam_width, t, -1])
        return curr_beam_width
