import tensorflow as tf
import copy
import random
from collections import OrderedDict

from icecaps.estimators.estimator_group import EstimatorGroup
from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.seq2seq_encoder_estimator import Seq2SeqEncoderEstimator
from icecaps.estimators.seq2seq_decoder_estimator import Seq2SeqDecoderEstimator
from icecaps.estimators.noise_layer import NoiseLayer


class SpaceFusionPreset(EstimatorGroup):

    def __init__(self, model_dir, params, config=None, scope="default"):
        self.core_encoder = Seq2SeqEncoderEstimator(model_dir, params, scope="core_encoder")
        self.ae_encoder = Seq2SeqEncoderEstimator(model_dir, params, scope="ae_encoder")
        self.noise = NoiseLayer(model_dir, params, scope="noise")
        self.decoder = Seq2SeqDecoderEstimator(model_dir, params, scope="decoder")
        self.core_model = EstimatorChain([self.core_encoder, self.decoder], model_dir, params, scope="core")
        self.noisy_core_model = EstimatorChain([self.core_encoder, self.noise, self.decoder], model_dir, params, scope="noisy_core")
        self.autoencoder = EstimatorChain([self.ae_encoder, self.noise, self.decoder], model_dir, params, scope="ae")
        self.loss_balance = [1.0, 1.0, 1.0, 1.0]
        super().__init__([self.core_model, self.noisy_core_model, self.autoencoder], model_dir, params, config=config, scope=scope)

    def set_loss_balance(self, loss_balance):
        self.loss_balance = loss_balance

    def dec_loss(self, z_conv, z_resp, interp_ratio):
        interp_features = dict()
        for field in z_conv.predictions:
            if field == "outputs":
                interp_features["inputs"] = interp_ratio * z_conv.predictions["outputs"][:,-1:,:] + (1 - interp_ratio) * z_resp.predictions["outputs"][:,-1:,:]
            elif field == "state":
                if isinstance(z_conv.predictions["state"], tuple):
                    tmp_state = []
                    for i in range(len(z_conv.predictions["state"])):
                        tmp_state.append(interp_ratio * z_conv.predictions["state"][i] + (1 - interp_ratio) * z_resp.predictions["state"][i])
                    interp_features["state"] = type(z_conv.predictions["state"])(*tmp_state)
                else:
                    interp_features["state"] = interp_ratio * z_conv.predictions["state"] + (1 - interp_ratio) * z_resp.predictions["state"]
            elif field == "inputs":
                interp_features["original_inputs"] = z_conv.predictions[field]
            else:
                interp_features[field] = z_conv.predictions[field]
        loss_interp = self.decoder._model_fn(interp_features, self.mode, self.params).loss
        return loss_interp, interp_features

    def sqrt_mse(self, z_conv, z_resp):
        a = z_conv.predictions["outputs"][:,-1:,:]
        b = z_resp.predictions["outputs"][:,-1:,:]
        d = tf.sqrt(tf.reduce_mean(tf.squared_difference(a, b)))
        return d

    def batch_dist_cross(self, z_conv, z_resp, cap=float("inf")):
        a = z_conv.predictions["outputs"][:,-1:,:]
        b = tf.expand_dims(z_resp.predictions["outputs"][:,-1,:], 0)
        d = tf.sqrt(tf.maximum(0.0, tf.reduce_mean(tf.squared_difference(a, b), 2)))
        d = tf.minimum(cap, d)
        return tf.reduce_sum(d) / tf.maximum(1.0, tf.cast(tf.shape(a)[0] * (tf.shape(a)[0] - 1), tf.float32))

    def batch_dist(self, z_pred, cap=0.3):
        return self.batch_dist_cross(z_pred, z_pred, cap=cap)

    def _model_fn(self, features, mode, params):
        self.extract_args(features, mode, params)
        if mode != tf.estimator.ModeKeys.TRAIN:
            return super()._model_fn(features, mode, params)

        conv_features = self.filter_features(features, "conv")
        z_conv = self.noisy_core_model.calculate_subchain(conv_features, tf.estimator.ModeKeys.PREDICT, params, -1)
        resp_features = copy.copy(conv_features)
        resp_features["inputs"] = resp_features["targets"]
        z_resp = self.autoencoder.calculate_subchain(resp_features, tf.estimator.ModeKeys.PREDICT, params, -1)

        self.loss  = self.loss_balance[0] * self.dec_loss(z_conv, z_resp, 1.0)[0]
        self.loss += self.loss_balance[1] * self.dec_loss(z_conv, z_resp, 0.0)[0]
        self.loss += self.loss_balance[2] * self.dec_loss(z_conv, z_resp, random.random())[0]
        self.loss += self.loss_balance[3] * (
            self.sqrt_mse(z_conv, z_resp) - self.batch_dist(z_conv) - self.batch_dist(z_resp))
        
        self.build_optimizer()
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=self.loss, train_op=self.train_op)

    def train(self, _input_fn, hooks=[], steps=None, max_steps=None, saving_listeners=None, logging_freq=10):
        return super().train(_input_fn, None, hooks, steps, max_steps, saving_listeners, logging_freq)

    def evaluate(self, _input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        return super().evaluate(_input_fn, self.core_model, steps, hooks, checkpoint_path, name)

    def predict(self, _input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        return super().predict(_input_fn, self.core_model, predict_keys, hooks, checkpoint_path, yield_single_examples)
