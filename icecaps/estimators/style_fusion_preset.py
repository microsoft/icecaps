import tensorflow as tf
import copy
import random
from collections import OrderedDict

from icecaps.estimators.estimator_group import EstimatorGroup
from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.seq2seq_encoder_estimator import Seq2SeqEncoderEstimator
from icecaps.estimators.seq2seq_decoder_estimator import Seq2SeqDecoderEstimator
from icecaps.estimators.noise_layer import NoiseLayer
from icecaps.estimators.space_fusion_preset import SpaceFusionPreset


class StyleFusionPreset(SpaceFusionPreset):

    def __init__(self, model_dir, params, config=None, scope="default"):
        super().__init__(model_dir, params, config=config, scope=scope)
        self.loss_balance = [0.67, 0.33, 0.5, 0.5, 0.5, 0.25, 0.25]

    def dec_loss_interp(self, z_resp, z_nonc, interp_ratio):
        loss_interp, interp_features = self.dec_loss(z_resp, z_nonc, interp_ratio)
        loss_interp = interp_ratio * loss_interp
        interp_features["targets"] = z_resp.predictions["targets"]
        loss_interp += (1 - interp_ratio) * self.decoder._model_fn(interp_features, self.mode, self.params).loss
        return loss_interp

    def _model_fn(self, features, mode, params):
        self.extract_args(features, mode, params)
        if mode != tf.estimator.ModeKeys.TRAIN:
            return super()._model_fn(features, mode, params)

        conv_features = self.filter_features(features, "conv")
        z_conv = self.noisy_core_model.calculate_subchain(conv_features, tf.estimator.ModeKeys.PREDICT, params, -1)
        resp_features = copy.copy(conv_features)
        resp_features["inputs"] = resp_features["targets"]
        z_resp = self.autoencoder.calculate_subchain(resp_features, tf.estimator.ModeKeys.PREDICT, params, -1)
        nonc_features = self.filter_features(features, "nonc")
        z_nonc = self.autoencoder.calculate_subchain(nonc_features, tf.estimator.ModeKeys.PREDICT, params, -1)

        self.loss  = self.loss_balance[0] * self.dec_loss(z_conv, z_resp, random.random())[0]
        self.loss += self.loss_balance[1] * self.dec_loss_interp(z_resp, z_nonc, random.random())
        self.loss += self.loss_balance[2] * self.sqrt_mse(z_conv, z_resp)
        self.loss += self.loss_balance[3] * self.batch_dist_cross(z_conv, z_nonc)
        self.loss += self.loss_balance[4] * self.batch_dist(z_conv)
        self.loss += self.loss_balance[5] * self.batch_dist(z_resp)
        self.loss += self.loss_balance[6] * self.batch_dist(z_nonc)

        self.build_optimizer()
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=self.loss, train_op=self.train_op)

