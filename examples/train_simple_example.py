import tensorflow as tf

from icecaps.io.data_source import DataSource
from icecaps.estimators.seq2seq_estimator import Seq2SeqEstimator
from icecaps.estimators.transformer_estimator import TransformerEstimator
from icecaps.util.vocabulary import Vocabulary
import icecaps.decoding.decoding as decoding
import icecaps.util.util as util


tf.app.flags.DEFINE_string('model_dir', './models/simple_example', 'Path to save model checkpoints')
tf.app.flags.DEFINE_boolean('clean_model_dir', False, 'Boolean to re-initialize model dir')
tf.app.flags.DEFINE_string('params_file', './dummy_params/simple_example_seq2seq.params', 'Newline-delimited parameters file')
tf.app.flags.DEFINE_integer('train_batches', 10000, 'Number of batches to train for')
tf.app.flags.DEFINE_string('train_file', './dummy_data/paired.tfrecord', 'TFRecord file for seq2seq training')
tf.app.flags.DEFINE_string('test_file', './dummy_data/paired.tfrecord', 'TFRecord file for seq2seq testing')
tf.app.flags.DEFINE_integer('batch_size', 128, "Batch size.")
tf.app.flags.DEFINE_boolean('interactive', True, "Enables command-line interactive decoding.")

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    '''
    This is a simple example of how to build an Icecaps training script, and is essentially
    the "Hello World" of Icecaps. Icecaps training scripts follow a basic five-phase pattern
    that we describe here. We train a basic model on the paired data stored in
    dummy_data/paired_personalized.tfrecord. For information on how to build TFRecords
    from text data files, please see data_processing_example.py.
    '''

    print("Loading hyperparameters..")
    # The first phase is to load hyperparameters from a .params file. These files follow a
    # simple colon-delimited format (e.g. see dummy_params/simple_example_seq2seq.params).
    params = util.load_params(FLAGS.params_file)

    print("Building model..")
    # Second, we build our architecture based on our loaded hyperparameters. Our architecture
    # here is very basic: we use a simple LSTM-based seq2seq model. For information on more
    # complex architectures, wee train_persona_mmi_example.py.
    model_dir = FLAGS.model_dir
    if FLAGS.clean_model_dir:
        util.clean_model_dir(model_dir)
    model_cls = Seq2SeqEstimator

    # Every estimator expects a different set of hyperparmeters. If you set use_default_params
    # to True in your .params file, the estimator will employ default values for any unspecified
    # hyperparameters. To view the list of hyperparmeters with default values, you can run the
    # class method list_params(). E.g. you can open a Python session and run
    # Seq2SeqEstimator.list_params() to view what hyperparameters our seq2seq estimator expects.
    model = model_cls(model_dir, params)

    print("Getting sources..")
    # Third, we set up our data sources. DataSource objects allow you to build input_fns that
    # efficiently feed data into the training pipeline from TFRecord files. In our simple example,
    # we set up two data sources: one for training and one for testing.

    # TFRecords are created with name variables per data point. You must create a fields dictionary
    # to tell the DataSource which variables to load and what their types are.
    fields = {"train/inputs": "int", "train/targets": "int"}
    train_source = DataSource(FLAGS.train_file, fields)
    test_source = DataSource(FLAGS.test_file, fields)

    # Then, you must create a field_map dictionary to tell your estimator how to map the TFRecord's
    # variable names to the names expected by the estimator. While this may seem like unnecessary
    # overhead in this simple example, it provides useful flexibility in more complex scenarios.
    field_map = {"inputs": "train/inputs", "targets": "train/targets"}

    # Finally, build input_fns from your DataSources.
    train_input_fn = train_source.get_input_fn(
        "train_in", field_map, None, FLAGS.batch_size)  # None lets our input_fn run for an unbounded
                                                        # number of epochs.
    test_input_fn = test_source.get_input_fn(
        "test_in", field_map, 1, FLAGS.batch_size)      # For testing, we only want to run the input_fn
                                                        # for one epoch instead.

    print("Processing model..")
    # Fourth, we pipe our input_fns through our model for training and evaluation.                                                
    model.train(train_input_fn, steps=FLAGS.train_batches)
    model.evaluate(test_input_fn)

    if FLAGS.interactive:
        print("Interactive decoding...")
        # Fifth, you may optionally set up an interactive session to test your system by directly
        # engaging with it.
        vocab = Vocabulary(fname=params["vocab_file"])
        decoding.cmd_decode(model, vocab)


if __name__ == '__main__':
    tf.app.run()
