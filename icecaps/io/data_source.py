import tensorflow as tf
import numpy as np
import random
from tensorflow.python.data import TFRecordDataset

from icecaps.util.vocabulary import Vocabulary


class DataSource:
    def __init__(self, fname, fields, vocab=None):
        self.fname = fname
        self.parse_fields(fields)
        self.input_fns = dict()
        self.vocab = vocab if vocab is not None else Vocabulary()

    def parse_fields(self, fields):
        self.fields = dict()
        for field in fields:
            if fields[field] == "int":
                self.fields[field] = tf.VarLenFeature(tf.int64)
            elif fields[field] == "float":
                self.fields[field] = tf.VarLenFeature(tf.float32)
            else:
                raise ValueError(
                    "Type " + str(fields[field]) + " for field " + str(field) + " is not supported.")

    def decode_record(self, record):
        data_items_to_decoders = {
            field: tf.contrib.slim.tfexample_decoder.Tensor(field)
            for field in self.fields
        }
        decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
            self.fields, data_items_to_decoders)
        decode_items = list(data_items_to_decoders)
        decoded = decoder.decode(record, items=decode_items)
        return dict(zip(decode_items, decoded))

    def tfrecord2iter(self, num_epochs, batch_size):
        dataset = TFRecordDataset(self.fname)
        dataset = dataset.map(lambda r: self.decode_record(r))
        dataset = dataset.repeat(num_epochs)
        padded_shapes = dict()
        padding_values = dict()
        for field in self.fields:
            padded_shapes[field] = [None]
            if self.fields[field].dtype == tf.int64:
                padding_values[field] = tf.constant(
                    self.vocab.end_token_id, tf.int64)
            elif self.fields[field].dtype == tf.float32:
                padding_values[field] = tf.constant(0.0, tf.float32)
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def get_input_fn(self, key=None, field_map=None, num_epochs=1, batch_size=128):
        if field_map is not None:
            def _input_fn():
                next_elem = self.tfrecord2iter(
                    num_epochs, batch_size).get_next()
                features = {key: next_elem[field_map[key]]
                            for key in field_map}
                return features, None
            if key is not None:
                self.input_fns[key] = _input_fn
        return self.input_fns[key]

    @staticmethod
    def group_input_fns(scope_ls, input_fn_ls):
        def _input_fn():
            features = dict()
            for i in range(len(input_fn_ls)):
                subfeatures, _ = input_fn_ls[i]()
                prefix = scope_ls[i] + "/"
                for key in subfeatures:
                    features[prefix + key] = subfeatures[key]
            return features, None
        return _input_fn

    @staticmethod
    def shuffle_input_fns(input_fn_ls, balance=None):
        if balance is None:
            balance = [1.0] * len(input_fn_ls)
        total = 0.0
        for elem in balance:
            total += elem
        norm_balance = [elem / total for elem in balance]

        def _input_fn():
            rng = random.random()
            threshold = 0.0
            for i in range(len(norm_balance)):
                threshold += norm_balance[i]
                if rng < threshold:
                    return input_fn_ls[i]()
            return input_fn_ls[-1]()
        return _input_fn
