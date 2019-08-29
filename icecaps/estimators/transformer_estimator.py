import tensorflow as tf
import copy

from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.transformer_encoder_estimator import TransformerEncoderEstimator
from icecaps.estimators.transformer_decoder_estimator import TransformerDecoderEstimator


class TransformerEstimator(EstimatorChain):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope="", is_mmi_model=False):
        self.encoder = TransformerEncoderEstimator(
            model_dir, params, config, scope=scope+"/encoder")
        self.decoder = TransformerDecoderEstimator(
            model_dir, params, config, scope=scope+"/decoder", is_mmi_model=is_mmi_model)
        super().__init__([self.encoder, self.decoder],
                         model_dir, params, config, scope)
