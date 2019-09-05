import tensorflow as tf

from icecaps.decoding.beam_search_decoder_custom import BeamSearchDecoder
from icecaps.decoding.dynamic_decoder_custom import dynamic_decode
from icecaps.estimators.seq2seq_decoder_estimator import Seq2SeqDecoderEstimator


class PersonaSeq2SeqDecoderEstimator(Seq2SeqDecoderEstimator):
    ''' Extends vanilla seq2seq to include speaker embeddings for multiple personas. '''

    def _model_fn(self, features, mode, params):
        output = super()._model_fn(features, mode, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            if hasattr(self, "speakers"):
                output.predictions["speaker_ids"] = self.speakers
        return output

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["max_num_speakers"] = cls.make_param(64)
        expected_params["speaker_embed_dim"] = cls.make_param(32)
        expected_params["train_speaker_embeddings_only"] = cls.make_param(False)
        return expected_params

    def build_embeddings(self):
        super().build_embeddings()
        self.speaker_embeddings = tf.get_variable(
            name='speaker_embeddings', shape=[self.hparams.max_num_speakers, self.hparams.speaker_embed_dim])
        if self.hparams.speaker_embed_dim != self.hparams.hidden_units:
            projection = tf.get_variable(
                name='speaker_embed_proj', shape=[self.hparams.speaker_embed_dim, self.hparams.hidden_units])
            self.speaker_embeddings = self.speaker_embeddings @ projection

    def embed_sparse_to_dense(self, sparse):
        with tf.variable_scope('embed_sparse_to_dense', reuse=tf.AUTO_REUSE):
            tokens_dense = tf.nn.embedding_lookup(self.token_embeddings, sparse)
            speakers_sparse = tf.cast(self.features["speaker_ids"], tf.int32)
            speakers_dense = tf.nn.embedding_lookup(
                self.speaker_embeddings, speakers_sparse)
            dense = tokens_dense + speakers_dense
        return dense

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

            sort_ids = tf.argsort(scores, direction="DESCENDING", stable=True, axis=0)
            scores = tf.gather_nd(scores, sort_ids)
            hypotheses = tf.gather_nd(hypotheses, sort_ids)
            input_query_ids = tf.gather_nd(input_query_ids, sort_ids)

            sort_ids = tf.argsort(input_query_ids, direction="ASCENDING", stable=True, axis=0)
            scores = tf.gather_nd(scores, sort_ids)
            hypotheses = tf.gather_nd(hypotheses, sort_ids)
            input_query_ids = tf.gather_nd(input_query_ids, sort_ids)

            speakers = tf.gather_nd(tf.convert_to_tensor(self.features["speaker_ids"]), input_query_ids)
            input_queries = tf.gather_nd(tf.convert_to_tensor(self.features["original_inputs"]), input_query_ids)

            self.rt_hypotheses = tf.identity(hypotheses)
            self.inputs_pred = tf.identity(input_queries)
            self.speakers = tf.identity(speakers)
            self.scores = tf.identity(scores)


    def build_optimizer(self):
        super().build_optimizer([self.speaker_embeddings]
                               if self.hparams.train_speaker_embeddings_only else None)
