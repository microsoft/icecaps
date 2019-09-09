import tensorflow as tf

from icecaps.io.data_source import DataSource
from icecaps.estimators.seq2seq_estimator import Seq2SeqEstimator
from icecaps.estimators.transformer_estimator import TransformerEstimator
from icecaps.util.vocabulary import Vocabulary
import icecaps.decoding.decoding as decoding
import icecaps.util.util as util


tf.app.flags.DEFINE_string('model_dir', './models/simple_example', 'Path to save model checkpoints')
tf.app.flags.DEFINE_boolean('clean_model_dir', False, 'Boolean to re-initialize model dir')
tf.app.flags.DEFINE_string('model_cls', 'seq2seq', 'Which model to use, among: transformer, seq2seq')
tf.app.flags.DEFINE_string('params_file', './dummy_params/simple_example_seq2seq.params', 'Newline-delimited parameters file')
tf.app.flags.DEFINE_integer('train_batches', 10000, 'Number of batches to train for')
tf.app.flags.DEFINE_string('train_file', './dummy_data/paired.tfrecord', 'TFRecord file for seq2seq training')
tf.app.flags.DEFINE_string('test_file', './dummy_data/paired.tfrecord', 'TFRecord file for seq2seq testing')
tf.app.flags.DEFINE_integer('batch_size', 128, "Batch size")
tf.app.flags.DEFINE_boolean('interactive', True, "Enables interactive decoding. Choose between off, cmd, gui")

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    print("Loading hyperparameters..")
    params = util.load_params(FLAGS.params_file)

    print("Building model..")
    model_dir = FLAGS.model_dir
    if FLAGS.clean_model_dir:
        util.clean_model_dir(model_dir)
    if FLAGS.model_cls == "transformer":
        model_cls = TransformerEstimator
    elif FLAGS.model_cls == "seq2seq":
        model_cls = Seq2SeqEstimator
    else:
        raise ValueError("Model class not supported.")
    model = model_cls(model_dir, params)

    print("Getting sources..")
    fields = {"train/inputs": "int", "train/targets": "int"}
    train_source = DataSource(FLAGS.train_file, fields)
    test_source = DataSource(FLAGS.test_file, fields)

    field_map = {"inputs": "train/inputs", "targets": "train/targets"}
    train_input_fn = train_source.get_input_fn(
        "train_in", field_map, None, FLAGS.batch_size)
    test_input_fn = test_source.get_input_fn(
        "test_in", field_map, 1, FLAGS.batch_size)

    print("Processing model..")
    model.train(train_input_fn, steps=FLAGS.train_batches)
    model.evaluate(test_input_fn)

    if FLAGS.interactive:
        print("Interactive decoding...")
        vocab = Vocabulary(fname=params["vocab_file"])
        decoding.cmd_decode(model, vocab)


if __name__ == '__main__':
    tf.app.run()
