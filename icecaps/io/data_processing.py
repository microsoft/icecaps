import os
import sys
import random
import tensorflow as tf
import copy
import re
from collections import defaultdict, OrderedDict

from icecaps.util.vocabulary import Vocabulary


class DataHeader:
    def __init__(self, name, data_type, vocab_file=None, vocab_mode="read"):
        self.name = name
        self.data_type = data_type
        self.vocab_file = vocab_file
        self.vocab_mode = vocab_mode


class DataProcessor:
    # sorting/shuffling
    # byte pair encoding
    # trait grounding

    def __init__(self, in_files, headers):
        self.in_files = in_files
        if isinstance(self.in_files, str):
            self.in_files = [self.in_files]
        self.headers = headers
        if isinstance(self.headers, DataHeader):
            self.headers = [self.headers]
        self.vocabs = dict()

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def lowercase(line):
        return line.lower()

    @staticmethod
    def tokenize_punctuation(line):
        line = re.sub(r"([^ ])(\.\.\.|''|``|!!|\.|!|,|\?|\)|\(|'|:|&) ", r"\1 \2 ", line)
        line = re.sub(r"([^ ])(\.\.\.|''|``|!!|\.|!|,|\?|\)|\(|'|:|&)$", r"\1 \2", line)
        line = re.sub(r"^([(\.\.\.)('')(``)(!!)\.!,\?\)\(':&])([^ ])", r"\1 \2", line)
        line = re.sub(r" ([(\.\.\.)('')(``)\(':&])([^ ])", r"\1 \2", line)
        return line

    @staticmethod
    def tokenize_capitalization(line):
        line = re.sub(r"([^ ])(n't|'m|'ll|'s|'d|'re|'ve)", r"\1 \2", line)
        return line

    @staticmethod
    def basic_preprocess(line):
        line = line.lower()
        line = DataProcessor.tokenize_punctuation(line)
        line = DataProcessor.tokenize_capitalization(line)
        return line    

    @staticmethod
    def length_limit(line, limit=25):
        if len(line.split(" ")) > limit:
            return None
        return line

    @staticmethod
    def extract_context(line, turns=1, eos_token='eos'):
        tokens = line.split(" ")
        try:
            eos_indices = [i for i, x in enumerate(
                tokens) if x == eos_token]
            if turns >= len(eos_indices):
                turn = 0
            start_idx = eos_indices[-turns]
        except ValueError as e:
            start_idx = 0
        return " ".join(tokens[start_idx:])

    @staticmethod
    def pad_context(line, turns=3, eos_token='eos'):
        tokens = line.split(" ")
        num_context = 0
        for token in tokens:
            if token == eos_token:
                num_context += 1
        if turns > num_context:
            tokens = [eos_token] * (turns - num_context) + tokens
        return " ".join(tokens)

    def row_gen_single_file(self, in_file):
        if isinstance(in_file, str):
            with open(in_file, "r", encoding="utf8") as in_f:
                for line in in_f:
                    yield line.strip().split("\t")
        elif isinstance(in_file, tuple) or isinstance(in_file, list):
            in_fs = []
            for i in range(len(self.headers)):
                in_fs.append(open(in_file[i], "r", encoding="utf8"))
            try:
                for lines in zip(*in_fs):
                    yield [l.strip() for l in lines]
            finally:
                for i in range(len(self.headers)):
                    in_fs[i].close()
        else:
            raise ValueError("in_file of type " + str(type(in_file)) + " not supported.")

    def row_gen(self):
        for in_file in self.in_files:
            yield from self.row_gen_single_file(in_file)

    def process(self, pipeline, line):
        for fn in pipeline:
            if line is None:
                break
            line = fn(line)
        return line

    def process_row(self, pipeline, row):
        break_flag = False
        for i in range(len(row)):
            if self.headers[i].data_type == "text" and pipeline is not None:
                if isinstance(pipeline, list) or isinstance(pipeline, tuple):
                    row[i] = self.process(pipeline, row[i])
                elif isinstance(pipeline, dict):
                    row[i] = self.process(pipeline[self.headers[i].name], row[i])
                else:
                    raise ValueError("Pipeline type " + str(type(pipeline)) + " is not supported.")
            if row[i] is None:
                break_flag = True
                break
        return not break_flag

    def print_lines_processed(self, line_ctr):
        line_ctr += 1
        if line_ctr % 10000 == 0:
            print("\t" + str(line_ctr) + " lines processed..")
        return line_ctr

    def build_vocab_files(self, count_cutoff=0):
        print("Building vocabularies..")
        read_only = True
        self.vocabs = dict()
        for i in range(len(self.headers)):
            vocab_ = self.headers[i].vocab_file
            mode = self.headers[i].vocab_mode
            if ((vocab_ is not None) and 
                (vocab_ not in self.vocabs) and 
                (mode != "read")):
                read_only = False
                if mode == "write":
                    self.vocabs[vocab_] = Vocabulary()
                elif mode == "append":
                    self.vocabs[vocab_] = Vocabulary(fname=vocab_)
                else:
                    raise ValueError("Vocab mode " + str(mode) + " not supported.")
            elif mode == "read":
                self.vocabs[vocab_] = Vocabulary(fname=vocab_)
        if read_only:
            return
        line_ctr = 0
        for row in self.row_gen():
            for i in range(len(row)):
                vocab_ = self.headers[i].vocab_file
                if vocab_ in self.vocabs:
                    self.vocabs[vocab_].tokenize(row[i], fixed_vocab=False)
            line_ctr = self.print_lines_processed(line_ctr)
        for vocab_ in self.vocabs:
            if count_cutoff >= 0:
                self.vocabs[vocab_].count_cutoff(count_cutoff)
            with open(vocab_, "w", encoding="utf8") as vocab_f:
                for word in self.vocabs[vocab_].words:
                    vocab_f.write(word + "\n")
        for i in range(len(self.headers)):
            self.headers[i].vocab_mode = "read"

    def build_byte_pair_encodings(self, out_file, bpe_size=1000):
        self.build_vocab_files()
        print("Constructing byte pair encodings..")
        all_bpe_vocabs = dict()
        for vocab_ in self.vocabs:
            word_encodings = dict()
            bpe_vocab = OrderedDict()
            for word in self.vocabs[vocab_].words:
                if word not in self.vocabs[vocab_].special_tokens:
                    word_encodings[word] = list(word) + ['</EOW>']
                    for element in word_encodings[word]:
                        bpe_vocab[element] = True
            bigram_counts = defaultdict(int)
            bigrams_to_words = defaultdict(dict)
            for word in word_encodings:
                for i in range(len(word_encodings[word]) - 1):
                    key = word_encodings[word][i] + word_encodings[word][i+1]
                    bigram_counts[key] += self.vocabs[vocab_].word_counts[word]
                    bigrams_to_words[key][word] = True
            while len(bpe_vocab) < bpe_size:
                if len(bigram_counts) == 0:
                    break
                winner = max(bigram_counts, key=bigram_counts.get)
                bpe_vocab[winner] = True
                if len(bpe_vocab) % 100 == 0:
                    print(str(len(bpe_vocab)) + " byte pairs processed..")
                for word in bigrams_to_words[winner]:
                    i = 0
                    while i < (len(word_encodings[word]) - 1):
                        key = word_encodings[word][i] + word_encodings[word][i+1]
                        if key == winner:
                            try:
                                word_encodings[word] = word_encodings[word][0:i] + \
                                    [winner] + word_encodings[word][i+2:]
                                reconstructed = ""
                                for elem in word_encodings[word]:
                                    reconstructed += elem
                                if word + "</EOW>" != reconstructed:
                                    raise ValueError(str(word) + " doesn't match " + str(reconstructed))
                            except:
                                raise ValueError("Some error with " + str(word) + " and " + str(winner) + " and " + str(word_encodings[word]))
                            if i > 0:
                                key = word_encodings[word][i-1] + word_encodings[word][i]
                                bigram_counts[key] += self.vocabs[vocab_].word_counts[word]
                                bigrams_to_words[key][word] = True
                            if (len(word_encodings[word]) - 1):
                                key = word_encodings[word][i-1] + word_encodings[word][i]
                                bigram_counts[key] += self.vocabs[vocab_].word_counts[word]
                                bigrams_to_words[key][word] = True
                        else:
                            i += 1
                del bigram_counts[winner]
                del bigrams_to_words[winner]
            new_vocab_fname = vocab_ + ".bpe"
            with open(new_vocab_fname, "w", encoding="utf8") as vocab_w:
                vocab_w.write("_END\n")
                vocab_w.write("_GO\n")
                vocab_w.write("_UNK\n")
                for word in bpe_vocab:
                    vocab_w.write(word + "\n")
        length_headers = OrderedDict()
        for i in range(len(self.headers)):
            if self.headers[i].vocab_file is not None:
                self.headers[i].vocab_file += ".bpe"
                length_headers[self.headers[i].name] = DataHeader(self.headers[i].name + "/_length", "int")
        for header in length_headers:
            self.headers.append(length_headers[header])
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            for row in self.row_gen():
                row_extension = []
                for i in range(len(row)):
                    vocab_ = self.headers[i].vocab_file
                    if vocab_ is not None:
                        row_extension.append(len(row[i]))
                        new_elem = ""
                        for word in row[i].strip().split():
                            for subword in word_encodings[word]:
                                new_elem += subword + " "
                        row[i] = new_elem
                row += row_extension
                out_f.write("\t".join([str(row[i]) for i in range(len(row))]) + "\n")
                line_ctr = self.print_lines_processed(line_ctr)
        self.in_files = [out_file]

    def apply_byte_pair_encodings(self, out_file, max_lines=None):
        self.build_vocab_files()
        print("Applying byte pair encodings..")
        all_bpe_vocabs = dict()
        word_encodings = dict()
        for vocab_ in self.vocabs:
            all_bpe_vocabs[vocab_] = Vocabulary(fname=vocab_)
            word_encodings[vocab_] = dict() 
        length_headers = OrderedDict()
        for i in range(len(self.headers)):
            if self.headers[i].vocab_file is not None:
                length_headers[self.headers[i].name] = DataHeader(self.headers[i].name + "/_length", "int")
        for header_name in length_headers:
            self.headers.append(length_headers[header_name])
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            for row in self.row_gen():
                row_extension = []
                for i in range(len(row)):
                    vocab_ = self.headers[i].vocab_file
                    if vocab_ is not None:
                        row_extension.append(len(row[i].strip().split()))
                        new_elem = ""
                        for word in row[i].strip().split():
                            if word in word_encodings[vocab_]:
                                encoding = word_encodings[vocab_][word]
                            else:
                                encoding = list(word) + ["</EOW>"]
                                bigrams = dict()
                                for j in range(len(encoding) - 1):
                                    bigram = encoding[j] + encoding[j+1]
                                    if bigram in all_bpe_vocabs[vocab_].word2idx:
                                        bigrams[j] = all_bpe_vocabs[vocab_].word2idx[bigram]
                                while len(bigrams) > 0:
                                    bigrams_argmin = None
                                    for idx in bigrams:
                                        if bigrams_argmin is None or bigrams[idx] < bigrams[bigrams_argmin]:
                                            bigrams_argmin = idx
                                    encoding = encoding[0:bigrams_argmin] + \
                                        [encoding[bigrams_argmin] + encoding[bigrams_argmin+1]] + encoding[bigrams_argmin+2:]
                                    bigrams = dict()
                                    for j in range(len(encoding) - 1):
                                        bigram = encoding[j] + encoding[j+1]
                                        if bigram in all_bpe_vocabs[vocab_].word2idx:
                                            bigrams[j] = all_bpe_vocabs[vocab_].word2idx[bigram]
                                word_encodings[vocab_][word] = encoding
                            for subword in encoding:
                                new_elem += subword + " "
                        row[i] = new_elem
                row += row_extension
                out_f.write("\t".join([str(row[i]) for i in range(len(row))]) + "\n")
                line_ctr = self.print_lines_processed(line_ctr)
                if max_lines is not None and line_ctr >= max_lines:
                    break
        self.in_files = [out_file]

    def k_tokens_per_line(self, out_file, k=20, hinge=None):
        # hinge is the name of the feature to split into k tokens per line
        if hinge is not None:
            hinge_id = -1
            for i in range(len(self.headers)):
                if self.headers[i].name == hinge:
                    hinge_id = i
                    break
            if hinge_id < 0:
                raise ValueError("Bad hinge: " + hinge + " is not the name of any DataHeader.")
        else:
            hinge_id = 0
        with open(out_file, "w", encoding="utf8") as out_f:
            out_line_buffer = []
            line_ctr = 0
            for row in self.row_gen():
                tokenized = row[hinge_id].strip().split() + ["<eos>"]
                for token in tokenized:
                    out_line_buffer.append(token)
                    if len(out_line_buffer) >= k:
                        out_f.write(' '.join(out_line_buffer) + '\n')
                        out_line_buffer = []
                line_ctr = self.print_lines_processed(line_ctr)
            if len(out_line_buffer) > 0:
                out_f.write(' '.join(out_line_buffer) + '\n')
        self.in_files = [out_file]

    def write_to_txt(self, out_file, pipeline=None, max_lines=None):
        print("Writing to txt..")
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            for row in self.row_gen():
                if not self.process_row(pipeline, row):
                    continue
                out_f.write("\t".join([str(row[i]) for i in range(len(row))]) + "\n")
                line_ctr = self.print_lines_processed(line_ctr)
                if max_lines is not None and line_ctr >= max_lines:
                    break
        self.in_files = [out_file]

    def write_to_tfrecord(self, out_file, pipeline=None, max_lines=None):
        print("Writing to TFRecord..")
        writer = tf.python_io.TFRecordWriter(out_file)
        line_ctr = 0
        for row in self.row_gen():
            if not self.process_row(pipeline, row):
                continue
            feature = dict()
            for i in range(len(row)):
                key_ = self.headers[i].name
                type_ = self.headers[i].data_type
                vocab_ = self.headers[i].vocab_file
                mode_ = self.headers[i].vocab_mode
                if type_ == "text":
                    if vocab_ not in self.vocabs:
                        if mode_ != "write":
                            self.vocabs[vocab_] = Vocabulary(fname=vocab_)
                        else:
                            self.vocabs[vocab_] = Vocabulary()
                    row[i] = self.vocabs[vocab_].tokenize(row[i], fixed_vocab=(mode_ == "read"))
                    feature[key_] = self.int64_feature(row[i])
                elif type_ == "int":
                    feature[key_] = self.int64_feature([int(row[i])])
                elif type_ == "float":
                    feature[key_] = self.float_feature([float(row[i])])
                else:
                    raise ValueError("Header type " + str(type_) + " not supported.")
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            line_ctr = self.print_lines_processed(line_ctr)
            if max_lines is not None and line_ctr >= max_lines:
                break
        writer.close()

