import tensorflow as tf
import copy

from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.seq2seq_encoder_estimator import Seq2SeqEncoderEstimator
from icecaps.estimators.persona_seq2seq_decoder_estimator import PersonaSeq2SeqDecoderEstimator


class PersonaSeq2SeqEstimator(EstimatorChain):

    def __init__(self, model_dir="/tmp", params=dict(), config=None, scope="", is_mmi_model=False):
        self.encoder = Seq2SeqEncoderEstimator(
            model_dir, params, config=config, scope=scope+"/encoder")
        self.decoder = PersonaSeq2SeqDecoderEstimator(
            model_dir, params, config=config, scope=scope+"/decoder", is_mmi_model=is_mmi_model)
        super().__init__([self.encoder, self.decoder],
                         model_dir, params, config, scope)

    @classmethod
    def list_params(cls, expected_params=None):
        print("Seq2Seq Encoder:")
        Seq2SeqEncoderEstimator.list_params(expected_params)
        print()
        print("Persona Seq2Seq Decoder:")
        PersonaSeq2SeqDecoderEstimator.list_params(expected_params)
        print()
