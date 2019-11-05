import tensorflow as tf

from icecaps.estimators.transformer_estimator import TransformerDecoderEstimator


class PersonaTransformerDecoderEstimator(TransformerDecoderEstimator):

    def _model_fn(self, features, mode, params):
        output = super()._model_fn(features, mode, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            if self.is_mmi_model:
                output.predictions["speaker_ids"] = self.features["speaker_ids"]
            else:
                output.predictions["speaker_ids"] = self.tiled_speakers
        return output

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["max_num_speakers"] = cls.make_param(64)
        expected_params["train_speaker_embeddings_only"] = cls.make_param(False)
        return expected_params

    def build_embeddings(self):
        super().build_embeddings()
        self.speaker_embeddings = tf.get_variable(
            name='speaker_embedding', shape=[self.hparams.max_num_speakers, self.hparams.d_model])

    def init_targets(self):
        super().init_targets()
        self.speakers_dense = tf.nn.embedding_lookup(
            params=self.speaker_embeddings, ids=self.features["speaker_ids"])
        speaker_start_tokens = self.speakers_dense # replace _GO token with speaker embedding
        self.inputs_dense = tf.concat(
            [speaker_start_tokens, tf.slice(self.inputs_dense - self.positions, [0,1,0], [-1,-1,-1])], 1) + self.positions

    def get_dense_start_tokens(self, effective_batch_size):
        start_tokens_dense = tf.nn.embedding_lookup(
            params=self.speaker_embeddings, ids=self.features["speaker_ids"])
        start_tokens_dense = tf.expand_dims(start_tokens_dense, 1)
        start_tokens_dense = tf.tile(
            start_tokens_dense, [1, effective_batch_size // self.batch_size, 1, 1])
        start_tokens_dense = tf.reshape(
            start_tokens_dense, [effective_batch_size, 1, self.hparams.d_model])
        return start_tokens_dense

    def build_optimizer(self):
        super().build_optimizer([self.speaker_embeddings]
                               if self.hparams.train_speaker_embeddings_only else None)

    def build_rt_decoder(self):
        curr_beam_width = super().build_rt_decoder()
        self.tiled_speakers = tf.reshape(
            tf.tile(tf.expand_dims(self.features["speaker_ids"], 1),
            [1, curr_beam_width, 1]), [self.batch_size*curr_beam_width, -1])