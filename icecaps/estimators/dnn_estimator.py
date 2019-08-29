import tensorflow as tf

from icecaps.estimators.abstract_icecaps_estimator import AbstractIcecapsEstimator


class DNNEstimator(AbstractIcecapsEstimator):

    def _model_fn(self, features, mode, params):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.extract_args(features, mode, params)
            self.build_model()
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.build_loss()
                self.build_optimizer()
                return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)
            self.predictions = {
                "inputs": self.features["inputs"],
                "state": self.logits,
                "outputs": self.features["inputs"],
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                print("Number of parameters: " +
                      str(self.get_num_model_params()))
                self.eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(self.features["targets"], self.outputs)
                }
                return tf.estimator.EstimatorSpec(mode, loss=self.loss, eval_metric_ops=self.eval_metric_ops)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=self.predictions)

    @classmethod
    def construct_expected_params(cls):
        expected_params = super().construct_expected_params()
        expected_params["act_f"] = cls.make_param('relu')
        expected_params["depth"] = cls.make_param(1)
        expected_params["hidden_dim"] = cls.make_param(16)
        expected_params["use_batch_norm"] = cls.make_param(True)
        expected_params["out_dim"] = cls.make_param(500)
        expected_params["regression"] = cls.make_param(False)
        return expected_params

    def extract_args(self, features, mode, params):
        super().extract_args(features, mode, params)
        self.set_act_f()

    def set_act_f(self):
        if self.hparams.act_f == 'tanh':
            self.act_f = tf.tanh
        elif self.hparams.act_f == 'sigmoid':
            self.act_f = tf.sigmoid
        elif self.hparams.act_f == 'relu':
            self.act_f = tf.nn.relu
        else:
            raise ValueError("Activation function " +
                             self.hparams.act_f + " not recognized. Aborting...")

    def build_model(self):
        with tf.name_scope('core'):
            signal = self.features["state"]
            for i in range(self.hparams.depth):
                signal = tf.layers.dense(signal, self.hparams.hidden_dim)
                if self.hparams.use_batch_norm:
                    signal = tf.layers.batch_normalization(
                        signal, training=self.mode == tf.estimator.ModeKeys.TRAIN, scale=(self.act_f != 'relu'))
                signal = tf.nn.dropout(self.act_f(signal), self.keep_prob)
            self.state = signal
        self.logits = tf.layers.dense(signal, self.hparams.out_dim)
        if self.hparams.regression:
            self.outputs = self.logits
        else:
            self.outputs = tf.argmax(input=self.logits, axis=-1)

    def build_loss(self):
        if self.hparams.regression:
            self.loss = tf.identity(tf.reduce_mean(tf.square(self.features["targets"] - self.state)))
        else:
            self.loss = tf.identity(tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                labels=self.features["targets"], logits=self.logits, weights=1.0)), "reported_loss")
