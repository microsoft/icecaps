import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.seq2seq import TrainingHelper
from tensorflow.python.layers.core import Dense

from icecaps.estimators.abstract_recurrent_estimator import AbstractRecurrentEstimator
from icecaps.decoding.basic_decoder_custom import BasicDecoder as MMIDecoder
from icecaps.decoding.beam_search_decoder_custom import BeamSearchDecoder
from icecaps.decoding.dynamic_decoder_custom import dynamic_decode
from icecaps.util.vocabulary import Vocabulary


class Seq2SeqDecoderEstimator(AbstractRecurrentEstimator):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope="", is_mmi_model=False):
        self.is_mmi_model = is_mmi_model
        super().__init__(model_dir, params, config, scope)


    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.batch_size = tf.shape(self.features["inputs"])[0]
            self.build_rnn()
            if mode == tf.estimator.ModeKeys.PREDICT:
                if self.is_mmi_model:
                    self.build_inputs()
                    self.build_mmi_decoder()
                    self.predictions = {
                        "inputs": tf.convert_to_tensor(self.features["original_inputs"]),
                        "targets": tf.convert_to_tensor(self.features["targets"]),
                        "scores": self.mmi_scores,
                    }
                    export_outputs = None
                else:
                    self.build_rt_decoder()
                    self.predictions = {
                        "inputs": self.inputs_pred,
                        "outputs": self.rt_hypotheses,
                        "scores": self.scores
                    }
                export_outputs = {
                    'predict_output': tf.estimator.export.PredictOutput(self.predictions)}
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions, export_outputs=export_outputs)
            self.build_inputs()
            self.build_train_decoder()
            self.build_loss()
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_optimizer()
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, train_op=self.train_op)
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = {
                    "gs_token_accuracy": tf.metrics.accuracy(
                        labels=self.targets_sparse, predictions=self.gs_hypotheses, weights=self.target_mask),
                    "program_accuracy": tf.metrics.accuracy(
                        labels=tf.zeros([self.batch_size], dtype=tf.int32),
                        predictions=tf.reduce_sum(tf.cast(self.target_mask, tf.int32) * tf.squared_difference(
                            self.targets_sparse, tf.cast(self.gs_hypotheses, tf.int32)), -1)),
                }
                return tf.estimator.EstimatorSpec(mode, loss=self.reported_loss, eval_metric_ops=self.eval_metric_ops)

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["use_attention"] = cls.make_param(False)
        expected_params["attention_input_feeding"] = cls.make_param(False)
        expected_params["attention_type"] = cls.make_param("luong")
        expected_params["shrink_vocab"] = cls.make_param(0)
        expected_params["repetition_allowance"] = cls.make_param(0.01)
        expected_params["repetition_penalty"] = cls.make_param(1.0)
        expected_params["post_repetition_penalty"] = cls.make_param(5.0)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        self.beam_search_decoding = tf.constant(
            self.mode == tf.estimator.ModeKeys.PREDICT and self.hparams.beam_width > 1 and not self.is_mmi_model)

    def build_inputs(self):
        super().build_inputs("targets")
        start_tokens_sparse = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * self.vocab.start_token_id
        start_tokens_dense = self.embed_sparse_to_dense(start_tokens_sparse)
        self.inputs_dense = tf.concat([start_tokens_dense, self.inputs_dense], axis=1)
        self.inputs_length += 1  # [batch_size]
        self.inputs_max_length += 1  # [batch_size, max_time_steps + 1]
        end_tokens_sparse = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * self.vocab.end_token_id
        self.targets_sparse = tf.concat([self.inputs_sparse, end_tokens_sparse], axis=1)
        self.target_mask = tf.sequence_mask(
            lengths=self.inputs_length, maxlen=self.inputs_max_length, dtype=tf.float32)

    def build_attention_mechanism(self):
        if self.hparams.attention_type == 'luong':
            attention_mechanism = seq2seq.LuongAttention(
                self.hparams.hidden_units, self.feedforward_inputs, self.feedforward_inputs_length)
        elif self.hparams.attention_type == 'bahdanau':
            attention_mechanism = seq2seq.BahdanauAttention(
                self.hparams.hidden_units, self.feedforward_inputs, self.feedforward_inputs_length,)
        else:
            raise ValueError(
                "Currently, the only supported attention types are 'luong' and 'bahdanau'.")

    def _attention_input_feeding(self, input_feed):
        if self.hparams.attention_input_feeding:
            self.attention_input_layer = Dense(self.hparams.hidden_units, name='attention_input_layer')
            return self.attention_input_layer(tf.concat([input_feed, attention], -1))
        else:
            return input_feed

    def build_attention_wrapper(self, final_cell):
        self.feedforward_inputs = tf.cond(
            self.beam_search_decoding, lambda: seq2seq.tile_batch(
                self.features["inputs"], multiplier=self.hparams.beam_width),
            lambda: self.features["inputs"])
        self.feedforward_inputs_length = tf.cond(
            self.beam_search_decoding, lambda: seq2seq.tile_batch(
                self.features["length"], multiplier=self.hparams.beam_width),
            lambda: self.features["length"])
        attention_mechanism = self.build_attention_mechanism()
        return AttentionWrapper(
            cell=final_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.hparams.hidden_units,
            cell_input_fn=self._attention_input_feeding,
            initial_cell_state=self.initial_state[-1] if self.hparams.depth > 1 else self.initial_state)

    def build_rnn(self):
        self.initial_state = tf.cond(
            self.beam_search_decoding, lambda: seq2seq.tile_batch(
                self.features["state"], self.hparams.beam_width),
            lambda: self.features["state"], name="initial_state")
        self.build_embeddings()
        cell_list = self.build_deep_cell(return_raw_list=True)
        if self.hparams.use_attention:
            cell_list[-1] = self.build_attention(cell_list[-1])
            if self.hparams.depth > 1:
                self.initial_state[-1] = final_cell.zero_state(batch_size=self.batch_size)
            else:
                self.initial_state = final_cell.zero_state(batch_size=self.batch_size)
        with tf.name_scope('rnn_cell'):
            self.cell = self.build_deep_cell(cell_list)
        self.output_layer = Dense(self.vocab.size(), name='output_layer')

    def build_train_decoder(self):
        with tf.name_scope('train_decoder'):
            training_helper = TrainingHelper(inputs=self.inputs_dense,
                                             sequence_length=self.inputs_length,
                                             time_major=False,
                                             name='training_helper')
            with tf.name_scope('basic_decoder'):
                training_decoder = BasicDecoder(cell=self.cell,
                                                helper=training_helper,
                                                initial_state=self.initial_state,
                                                output_layer=self.output_layer)
            with tf.name_scope('dynamic_decode'):
                (outputs, self.last_state, self.outputs_length) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.inputs_max_length))
                self.logits = tf.identity(outputs.rnn_output)
                self.log_probs = tf.nn.log_softmax(self.logits)
                self.gs_hypotheses = tf.argmax(self.log_probs, -1)

    def build_rt_decoder(self):
        self.build_embeddings()
        start_tokens_sparse = tf.ones(shape=[self.batch_size], dtype=tf.int32) * self.vocab.start_token_id
        with tf.name_scope('beamsearch_decoder'):
            rt_decoder = BeamSearchDecoder(cell=self.cell,
                                           embedding=self.embed_sparse_to_dense,
                                           start_tokens=start_tokens_sparse,
                                           end_token=self.vocab.end_token_id,
                                           initial_state=self.initial_state,
                                           beam_width=self.hparams.beam_width,
                                           output_layer=self.output_layer,
                                           skip_tokens_decoding=self.vocab.skip_tokens,
                                           shrink_vocab=self.hparams.shrink_vocab)
            (hypotheses, input_query_ids, scores) = dynamic_decode(
                decoder=rt_decoder,
                output_time_major=False,
                maximum_iterations=self.hparams.max_length,
                repetition=self.hparams.repetition_penalty)

            sort_ids = tf.argsort(
                scores, direction="DESCENDING", stable=True, axis=0)
            scores = tf.gather_nd(scores, sort_ids)
            hypotheses = tf.gather_nd(hypotheses, sort_ids)
            input_query_ids = tf.gather_nd(input_query_ids, sort_ids)

            sort_ids = tf.argsort(
                input_query_ids, direction="ASCENDING", stable=True, axis=0)
            scores = tf.gather_nd(scores, sort_ids)
            hypotheses = tf.gather_nd(hypotheses, sort_ids)
            input_query_ids = tf.gather_nd(input_query_ids, sort_ids)

            input_queries = tf.gather_nd(tf.convert_to_tensor(
                self.features["original_inputs"]), input_query_ids)
            self.rt_hypotheses = tf.identity(hypotheses)
            self.inputs_pred = tf.identity(input_queries)
            self.scores = tf.identity(scores)

    def build_mmi_decoder(self):
        with tf.name_scope('mmi_scorer'):
            training_helper = TrainingHelper(inputs=self.inputs_dense,
                                             sequence_length=self.inputs_length,
                                             time_major=False,
                                             name='mmi_training_helper')
            with tf.name_scope('mmi_basic_decoder'):
                training_decoder = MMIDecoder(cell=self.cell,
                                              helper=training_helper,
                                              initial_state=self.initial_state,
                                              output_layer=self.output_layer)
            with tf.name_scope('mmi_dynamic_decoder'):
                (outputs, self.last_state, self.outputs_length) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.inputs_max_length)

            self.scores_raw = tf.identity(
                tf.transpose(outputs.scores, [1, 2, 0]))
            targets = self.features["targets"]
            targets = tf.cast(targets, dtype=tf.int32)
            target_len = tf.cast(tf.count_nonzero(
                targets - self.vocab.end_token_id, -1), dtype=tf.int32)
            max_target_len = tf.reduce_max(target_len)
            pruned_targets = tf.slice(targets, [0, 0], [-1, max_target_len])

            index = (tf.range(0, max_target_len, 1)) * \
                tf.ones(shape=[self.batch_size, 1], dtype=tf.int32)
            row_no = tf.transpose(tf.range(
                0, self.batch_size, 1) * tf.ones(shape=(max_target_len, 1), dtype=tf.int32))
            indices = tf.stack([index, pruned_targets, row_no], axis=2)

            # Retrieve scores corresponding to indices
            batch_scores = tf.gather_nd(self.scores_raw, indices)
            self.mmi_scores = tf.reduce_sum(batch_scores, axis=1)
