import tensorflow as tf

from icecaps.estimators.abstract_transformer_estimator import AbstractTransformerEstimator
import icecaps.util.trees as trees


class TransformerEncoderEstimator(AbstractTransformerEstimator):

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()):
            self.extract_args(features, mode, params)
            self.build_embeddings()
            self.init_inputs()
            self.build_encoder()
            if mode == tf.estimator.ModeKeys.PREDICT:
                self.predictions = {
                    "inputs": self.features["inputs"],
                    "outputs": self.outputs,
                    "mask": self.mask,
                    "length": self.inputs_length,
                    "token_embeddings": self.token_embeddings,
                }
                if "targets_positions" in self.features:
                    self.predictions["outputs_positions"] = self.features["targets_positions"]
                for key in [k for k in self.features if (k == 'metadata' or k.startswith('metadata/'))]:
                    predictions[key] = self.features[key]
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_optimizer()
                raise NotImplementedError(
                    "Training not currently supported for transformer encoder.")
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = dict()
                raise NotImplementedError(
                    "Evaluation not currently supported for transformer encoder.")

    def build_layer(self, x, batch_size):
        signal = self.build_mha_sublayer(
            x, x, batch_size, enc_mask=self.mask, dec_mask=None)
        signal = self.build_ffn_sublayer(signal, self.d_ff)
        return signal

    def build_encoder(self):
        signal = self.inputs_dense  # self.build_layer_norm(self.inputs_dense)
        for i in range(self.hparams.depth):
            with tf.variable_scope("layer_" + str(i), reuse=tf.AUTO_REUSE) as scope:
                signal = self.build_layer(signal, self.batch_size)
        self.outputs = signal
