import os
import sys
import random
import tensorflow as tf
import copy
from collections import defaultdict

from icecaps.data_io.data_header import DataHeader
from icecaps.data_io.text_data_processor import TextDataProcessor
from icecaps.data_io.json_data_processor import JSONDataProcessor


def main(_):
    '''
    In this example, we present a simple example of our data processing tools. We will convert
    a personalized conversational text data set to TFRecords, the format TensorFlow requires
    for feeding in data efficiently. We will use the provided dummy data set found in
    dummy_data/paired_personalized.txt and dummy_data/unpaired_personalized.txt . 
    '''

    # DataHeaders tell the data processing system how to handle each kind of data.
    # We will need one DataHeader per column in the source file.
    input_header = DataHeader("for_tree", "tree", "dummy_data/for2lam.dic", "write")
    #speaker_header = DataHeader("train/speakers", "int")
    target_header = DataHeader("lam_tree", "tree", "dummy_data/for2lam.dic", "write")
    data_headers = [input_header, target_header]

    # TextDataProcessor processes the data according to the headers.
    # In this case, we first build the vocabulary files specified in the headers.
    # Both headers share a vocabulary file here since the input and target belong to the same vocabulary.
    # We then call write_to_tfrecord() to create our TFRecord.
    paired_data_proc = TextDataProcessor("dummy_data/paired_personalized.txt", data_headers)
    paired_data_proc.build_vocab_files()
    paired_data_proc.write_to_tfrecord("dummy_data/paired_personalized.tfrecord")

    # We set our header's vocab mode to "read" to reuse the vocabulary file for the unpaired text data.
    input_header.vocab_mode = "read"
    data_headers = [speaker_header, input_header]

    # Finally we create another TextDataProcessor to convert the unpaired text data according to the
    # vocabulary we built earlier.
    unpaired_data_proc = TextDataProcessor("dummy_data/unpaired_personalized.txt", data_headers)
    unpaired_data_proc.write_to_tfrecord("dummy_data/unpaired_personalized.tfrecord")



if __name__ == '__main__':
    tf.app.run()
