import os
import sys
import random
import tensorflow as tf
import copy
from collections import defaultdict, OrderedDict

from icecaps.data_io.data_header import DataHeader
from icecaps.data_io.abstract_data_processor import AbstractDataProcessor
from icecaps.util.vocabulary import Vocabulary


class TextDataProcessor(AbstractDataProcessor):
    def row_gen_single_file(self, in_file, line_shard_len=None):
        if isinstance(in_file, str):
            with open(in_file, "r", encoding="utf8") as in_f:
                for line in in_f:
                    segments = line.strip().split("\t")
                    if len(segments) > 1 or line_shard_len is None:
                        yield segments
                    else:
                        segment = segments[0]
                        while len(segment) > line_shard_len:
                            yield segment[:line_shard_len]
                            segment = segment[line_shard_len:]
                        if len(segment) > 0:
                            yield segment
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

    def concatenate_segments(self, row):
        return "\t".join([str(row[i]) for i in range(len(row))]) + "\n"

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

    def _sort_key(self, x, sort_order):
        def _length(y):
            if isinstance(y, str):
                return len(y.split(" "))
            elif isinstance(y, TreeNode):
                return len([node for node in y.depth_first_traversal()])
            else:
                raise ValueError("Invalid type.")
        return tuple([_length(x[self.fields2idx[h]]) for h in sort_order])

    def _sort_and_write_batch(self, out_file, sort_order, sort_chunk_size, shuffle_batch_size, merge_segments, num_subfiles):
        sort_key = lambda x: self._sort_key(x, sort_order)
        merge_segments.sort(key=sort_key)  # sort by tgt len then by src len
        if shuffle_batch_size is not None:
            num_shuffle_batches = sort_chunk_size // shuffle_batch_size
            batch_ids = list(range(num_shuffle_batches))
            random.shuffle(batch_ids)
            new_merge_segments = []
            for i in range(num_shuffle_batches):
                new_merge_segments += merge_segments[shuffle_batch_size *
                                                     batch_ids[i]: shuffle_batch_size * (batch_ids[i] + 1)]
            if sort_chunk_size % shuffle_batch_size != 0:
                new_merge_segments += merge_segments[shuffle_batch_size *
                                                     num_shuffle_batches:]
            merge_segments = new_merge_segments
        txt = open(out_file + "_merge_" + str(0) + "_sub_" +
                   str(num_subfiles) + ".txt", "w", encoding="utf8")
        for row in merge_segments:
            txt.write(self.concatenate_segments(row))
        txt.close()
        return txt

    def local_sort_and_shuffle(self, out_file, sort_order, sort_chunk_size=1024, keep_temp_files=False,
                               shuffle_batch_size=None, merge_down=True):
        print("Local sorting and shuffling..")
        num_subfiles = 0
        merge_segments = []
        line_ctr = 0
        text_merge_f = None
        for row in self.row_gen():
            merge_segments.append(row)
            line_ctr += 1
            if line_ctr % sort_chunk_size == 0:
                text_merge_f = self._sort_and_write_batch(
                    out_file, sort_order, sort_chunk_size, shuffle_batch_size, merge_segments, num_subfiles)
                num_subfiles += 1
                merge_segments = []
        if len(merge_segments) > 0:
            text_merge_f = self._sort_and_write_batch(
                out_file, sort_order, sort_chunk_size, shuffle_batch_size, merge_segments, num_subfiles)
            num_subfiles += 1
        if num_subfiles > 1 and num_subfiles % 2 == 1:
            text_merge_f = open(out_file + "_merge_" + str(0) +
                                "_sub_" + str(num_subfiles) + ".txt", "w", encoding="utf8")
            text_merge_f.close()
            num_subfiles += 1
        if merge_down:
            with open(out_file, "w", encoding="utf8") as out_f:
                for i in range(num_subfiles):
                    fname = out_file + "_merge_" + str(0) + "_sub_" + str(i) + ".txt"
                    with open(fname, "r", encoding="utf8") as out_sub_f:
                        for line in out_sub_f:
                            out_f.write(line)
                    if not keep_temp_files:
                        os.remove(fname)
            self.in_files = [out_file]
        else:
            return num_subfiles, text_merge_f

    # Performs mergesort on full dataset.
    def global_sort(self, out_file, sort_order, sort_chunk_size=1024, keep_temp_files=False):
        num_subfiles, text_merge_f = self.local_sort_and_shuffle(
            out_file, sort_order, sort_chunk_size, keep_temp_files, merge_down=False)
        print("Global sorting..")
        sort_key = lambda x: self._sort_key(x, sort_order)
        num_subfiles = num_subfiles // 2
        merge_ctr = 0
        while num_subfiles > 0:
            for i in range(num_subfiles):
                text_merge_f = open(out_file + "_merge_" + str(merge_ctr + 1) +
                                    "_sub_" + str(i) + ".txt", "w", encoding="utf8")
                txt_0 = open(out_file + "_merge_" + str(merge_ctr) +
                             "_sub_" + str(2*i) + ".txt", "r", encoding="utf8")
                txt_1 = open(out_file + "_merge_" + str(merge_ctr) +
                             "_sub_" + str(2*i + 1) + ".txt", "r", encoding="utf8")
                merge_segments = []
                line_0 = txt_0.readline()
                line_1 = txt_1.readline()
                flag_0 = flag_1 = True
                segments_0 = segments_1 = None
                while line_0 != "" and line_1 != "":
                    if flag_0:
                        segments_0 = line_0.strip().split("\t")
                        flag_0 = False
                    if flag_1:
                        segments_1 = line_1.strip().split("\t")
                        flag_1 = False
                    if sort_key(segments_0) <= sort_key(segments_1):
                        text_merge_f.write(self.concatenate_segments(segments_0))
                        line_0 = txt_0.readline()
                        flag_0 = True
                    else:
                        text_merge_f.write(self.concatenate_segments(segments_1))
                        line_1 = txt_1.readline()
                        flag_1 = True
                (txt, segments) = (txt_0, segments_0) if line_0 != "" else (txt_1, segments_1)
                if segments is not None:
                    merge_segments.append(segments)
                    text_merge_f.write(self.concatenate_segments(segments))
                line = txt.readline()
                while line != "":
                    text_merge_f.write(line)
                    line = txt.readline()
                txt_0.close()
                txt_1.close()
                if not keep_temp_files:
                    os.remove(txt_0.name)
                    os.remove(txt_1.name)
                text_merge_f.close()
            if num_subfiles > 1 and num_subfiles % 2 == 1:
                text_merge_f = open(out_file + "_merge_" + str(merge_ctr + 1) +
                                    "_sub_" + str(num_subfiles) + ".txt", "w", encoding="utf8")
                text_merge_f.close()
                num_subfiles += 1
            num_subfiles = num_subfiles // 2
            merge_ctr += 1
        if os.path.exists(out_file):
            os.remove(out_file)
        os.rename(text_merge_f.name, out_file)
        self.in_files = [out_file]
        return out_file

    def write_to_txt(self, out_file, pipeline=None, max_lines=None, line_gen=None):
        print("Writing to txt..")
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            if line_gen is None:
                line_gen = self.row_gen()
            for row in line_gen:
                if not self.process_row(row, pipeline):
                    continue
                out_f.write(self.concatenate_segments(row))
                line_ctr = self.print_lines_processed(line_ctr)
                if max_lines is not None and line_ctr >= max_lines:
                    break
        self.in_files = [out_file]

    def trait_ground(self, out_file, field, num_ungrounded=1, num_grounded=1, pipeline=None, max_lines=None, line_gen=None):
        print("Trait grounding..")
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            if line_gen is None:
                line_gen = self.row_gen()
            for row in line_gen:
                if not self.process_row(row, pipeline):
                    continue
                for i in range(num_ungrounded):
                    out_f.write(self.concatenate_segments(row))
                j = 0
                idx = self.fields2idx[field]
                for key in sorted(row[idx].split(' '),
                                  key=lambda x: self.vocabs[self.headers[idx].vocab_file].word_counts[x]):
                    if j >= num_grounded:
                        break
                    modified_row = copy.deepcopy(row)
                    modified_row[idx] = row[idx] + ' _GO ' + key
                    out_f.write(self.concatenate_segments(modified_row))
                    j += 1
                line_ctr = self.print_lines_processed(line_ctr)
                if max_lines is not None and line_ctr >= max_lines:
                    break
        self.in_files = [out_file]

    def _write_to_shards(self, write_fn, file_extension, out_prefix, pipeline=None, max_lines=None, num_shards=20, dispose_remainder=False):
        num_lines = self.count_lines()
        if max_lines is not None:
            num_lines = min(max_lines, self.count_lines())
        lines_per_shard = [num_lines // num_shards] * num_shards
        if not dispose_remainder:
            for i in range(num_lines % num_shards):
                lines_per_shard[i] += 1
        line_gen = self.row_gen()
        for i in range(num_shards):
            write_fn(out_prefix + ".shard_" + str(i) + file_extension, pipeline=pipeline, max_lines=lines_per_shard[i], line_gen=line_gen)

    def write_to_sharded_txts(self, out_prefix, pipeline=None, max_lines=None, num_shards=20, dispose_remainder=False):
        self._write_to_shards(self.write_to_txt, ".txt", out_prefix, pipeline, max_lines, num_shards, dispose_remainder)

    def write_to_sharded_tfrecords(self, out_prefix, pipeline=None, max_lines=None, num_shards=20, dispose_remainder=False):
        self._write_to_shards(self.write_to_tfrecord, ".tfrecord", out_prefix, pipeline, max_lines, num_shards, dispose_remainder)
