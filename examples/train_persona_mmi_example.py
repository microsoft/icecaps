import tensorflow as tf

import icecaps.util.trees as trees
import icecaps.util.util as util
import icecaps.decoding.decoding as decoding
from icecaps.util.vocabulary import Vocabulary
from icecaps.estimators.transformer_estimator import TransformerEstimator
from icecaps.estimators.transformer_encoder_estimator import TransformerEncoderEstimator
from icecaps.estimators.persona_seq2seq_estimator import PersonaSeq2SeqEstimator
from icecaps.estimators.seq2seq_estimator import Seq2SeqEstimator
from icecaps.estimators.seq2seq_encoder_estimator import Seq2SeqEncoderEstimator
from icecaps.estimators.estimator_chain import EstimatorChain
from icecaps.estimators.estimator_group import EstimatorGroup
from icecaps.io.data_source import DataSource


tf.app.flags.DEFINE_string('model_dir', './models/persona_mmi_example', 'Path to save model checkpoints')
tf.app.flags.DEFINE_boolean('clean_model_dir', False, 'Boolean to re-initialize model dir')
tf.app.flags.DEFINE_string('params_file', './dummy_params/persona_mmi_example.params', 'Newline-delimited parameters file')
tf.app.flags.DEFINE_integer('pretrain_batches', 1000, 'Number of epochs to pretrain primary model for')
tf.app.flags.DEFINE_integer('train_batches', 10000, 'Number of epochs to train multitask model for')
tf.app.flags.DEFINE_integer('mmi_batches', 10000, 'Number of epochs to train mmi model for')
tf.app.flags.DEFINE_string('train_file', './dummy_data/paired_personalized.tfrecord', 'TFRecord file for seq2seq training')
tf.app.flags.DEFINE_string('autoenc_file', './dummy_data/unpaired_personalized.tfrecord', 'TFRecord file for autoencoder training')
tf.app.flags.DEFINE_string('test_file', './dummy_data/paired_personalized.tfrecord', 'TFRecord file for seq2seq testing')
tf.app.flags.DEFINE_integer('batch_size', 128, "Batch size")
tf.app.flags.DEFINE_boolean('interactive', True, "Enables interactive decoding session.")
tf.app.flags.DEFINE_float(
    'lambda_balance', 0.6,
    'Tradeoff between diversity and structure (forward_score * (1 - lambda) + reverse_score * lambda). Range: [0,1]')

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    print("Loading parameters..")
    params = util.load_params(FLAGS.params_file)

    print("Building model..")
    model_dir = FLAGS.model_dir
    if FLAGS.clean_model_dir:
        util.clean_model_dir(model_dir)
    first_model = PersonaSeq2SeqEstimator(model_dir, params, scope="first")
    second_model_encoder = Seq2SeqEncoderEstimator(
        model_dir, params, scope="second_encoder")
    second_model = EstimatorChain(
        [second_model_encoder, first_model.decoder], model_dir, params, scope="second")
    mmi_model = PersonaSeq2SeqEstimator(
        model_dir, params, scope="mmi", is_mmi_model=True)
    model_group = EstimatorGroup(
        [first_model, second_model, mmi_model], model_dir, params, scope="group")

    print("Getting sources..")
    fields = {"train/inputs": "int", "train/targets": "int", "train/speakers": "int"}
    train_source = DataSource(FLAGS.train_file, fields)
    autoenc_source = DataSource(FLAGS.autoenc_file, fields)
    test_source = DataSource(FLAGS.test_file, fields)
    
    train_field_map = {"inputs": "train/inputs", "targets": "train/targets", "speaker_ids": "train/speakers"}
    autoenc_field_map = {"inputs": "train/inputs", "targets": "train/inputs", "speaker_ids": "train/speakers"}
    mmi_field_map = {"inputs": "train/targets", "targets": "train/inputs", "speaker_ids": "train/speakers"}

    paired_input_fn = train_source.get_input_fn("paired_in", train_field_map, None, FLAGS.batch_size)
    autoenc_input_fn = train_source.get_input_fn("autoenc_in", autoenc_field_map, None, FLAGS.batch_size)
    mmi_input_fn = train_source.get_input_fn("mmi_in", mmi_field_map, None, FLAGS.batch_size)
    train_input_fn = DataSource.group_input_fns(["first", "second", "mmi"], [paired_input_fn, autoenc_input_fn, mmi_input_fn])
    test_input_fn = test_source.get_input_fn("test_in", train_field_map, 1, FLAGS.batch_size)
    
    print("Processing models..")
    print("Pretraining primary model..")
    model_group.train(train_input_fn, first_model, steps=FLAGS.pretrain_batches)
    print("Multitask training..")
    model_group.train(train_input_fn, {"first": 1, "second": 1, "mmi": 0}, steps=FLAGS.train_batches)
    print("Training MMI model..")
    model_group.train(train_input_fn, mmi_model, steps=FLAGS.mmi_batches)
    print("Evaluating..")
    model_group.evaluate(test_input_fn, first_model)
    
    if FLAGS.interactive:
        print("Interactive decoding...")
        vocab = Vocabulary(fname=params["vocab_file"])
        decoding.cmd_decode(first_model, vocab, persona=True, mmi_component=mmi_model)


if __name__ == '__main__':
    tf.app.run()
