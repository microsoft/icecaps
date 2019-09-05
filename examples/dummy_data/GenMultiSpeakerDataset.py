# This script generates toy data to test speaker-based conversation modeling.
#
# E.g. parameters:
# python .\train_persona_seq2seq.py --model_dir ./models/Persona-64-NoAtten-256-128 --valid_file ./data_simple_persona/data_autoenc_valid.tfrecord --interactive --multitask --hidden_units 256 --token_embed_dim 128 --speaker_embed_dim 64 --max_num_speakers 20 --evaluate --decode --pretrain_epochs 5 --train_epochs 5
#

import argparse
import numpy as np
import math
import subprocess
import os

from icecaps.io.data_processing import DataProcessor, DataHeader

parser = argparse.ArgumentParser(
    description='Generates dummy data with a unique persona for the autoencoder data')
parser.add_argument('-po', '--paired_out', help='file to write the paired data to',
                    default='my_paired_personalized')
parser.add_argument('-uo', '--unpaired_out', help='file to write the unpaired data to',
                    default='my_unpaired_personalized')
parser.add_argument('-pvo', '--paired_validation_out', help='file to write the paired validation data to',
                    default='my_paired_personalized_validation')
parser.add_argument('-vo', '--vocab_out', help='file to write vocabulary file to',
                    default='my_vocab_personalized')
parser.add_argument('-pc', '--paired_count', help='repeat data to get minimum number of paired data entries',
                    default=10000, type=int)
parser.add_argument('-uc', '--unpaired_count',
                    help='repeat data to get minimum number of unpaired data entries', default=2000, type=int)
args = vars(parser.parse_args())

seed = 1
np.random.seed(seed)

questions = [
    "what are you doing for a living ?",
    "what is your favorite color ?",
    "where do you live ?",
    "how old are you ?",
    "how much money do you have ?",
    "how are you doing ?",
]

answer_templates = [  # Persona id 0               1               2               3                   4
    [["I am {}", "I work as {}", "{}"],                    ['an admin',
                                                            'an engineer', 'a lawyer',      'a social worker',  'a student']],
    [['{}', 'my favorite color is {}', 'i love {}'],       ['red',
                                                            'green',        'magenta',      'purple',           'orange']],
    [['I am from {}', 'I live in {}', 'in {}', 'near {}'], ['london',
                                                            'seattle',      'boston',       'barcelona',        'tacoma']],
    [['I am {}', '{} years old', 'about {}'],              [
        '25',          '42',           '51',           '67',               '21']],
    [['about {}', 'i think around {}', '{}'],              ['200 dollars',
                                                            '500 dollars',  '700 dollars',  '900 dollars',      '50 dollars']],
    [["I am feeling {}", 'I am {}'],                       ['good',
                                                            'great',        'awesome',      'terrible',         'awful']],
]

def write_file_data(file_path, data, replicate_to_count):
    with open(file_path, 'w', encoding="utf8") as f:
        print(file_path)
        count = 0
        while(count < replicate_to_count):
            f.writelines(data)
            count += len(data)

# Seq2Seq Data:

data = []
num_pure_personas = len(answer_templates[0][1])
num_questions = len(answer_templates)
num_possible_personas = num_pure_personas**num_questions
sample_prob = 0.001  # to ensure the final dataset size is pretty small
# per-question personality vector
personality = np.zeros(shape=(num_questions), dtype=int)
base = num_pure_personas
persona_id = 0

for p in range(num_possible_personas):
    for k in range(len(personality)):   # iterate through a base(num_pure_personas) space
        if (personality[k] == (base - 1)):
            personality[0:k+1] = 0
            continue
        personality[k] += 1
        break

    # sub-sample the persona space. Keep the pure personas though
    if (p > num_pure_personas and np.random.rand() < (1-sample_prob)):
        continue

    for i, question in enumerate(questions):
        for template in answer_templates[i][0]:
            ansIdx = personality[i] if p >= num_pure_personas else p
            data.append((question + "\t{}\t" + template +
                         "\n").format(persona_id, answer_templates[i][1][ansIdx]))

    persona_id += 1

np.random.shuffle(data)
write_file_data(args["paired_out"] + ".txt", data, args["paired_count"])


# Auto Enc Data:
autoenc_data = []
autoenc_valid_data = []
persona_id += 1  # grab the next id

# add two new unique personas. Note that the vocab is a subset of the main s2s dataset. See *NOTE below.
autoenc_answers = [['a lawyer',    'a student'],
                   ['green',       'magenta'],
                   ['portland',      'london'],
                   ['25',          '20'],
                   ['700 dollars', '50 dollars'],
                   ['great',       'good']]

# Use the same answer formats as the original personas.
autoenc_data = []
autoenc_valid_data = []
for i, question in enumerate(questions):
    for template in answer_templates[i][0]:
        entries = [('{}\t' + template + '\n').format(persona_id + ans_idx, ans)
                   for ans_idx, ans in enumerate(autoenc_answers[i])]
        autoenc_data.extend(entries)
        entries = [(question + "\t{}\t" + template + "\n").format(persona_id + ans_idx, ans)
                   for ans_idx, ans in enumerate(autoenc_answers[i])]
        autoenc_valid_data.extend(entries)
np.random.shuffle(autoenc_data)
write_file_data(args["paired_validation_out"] + ".txt", autoenc_valid_data, 1)
write_file_data(args["unpaired_out"] + ".txt", autoenc_data, args["unpaired_count"])


# Create the dict and tfrecord files

in_header = DataHeader("train/inputs", "text", args["vocab_out"] + ".dic", "write")
tgt_header = DataHeader("train/targets", "text", args["vocab_out"] + ".dic", "write")
spk_header = DataHeader("train/speakers", "int")

data_proc = DataProcessor(args["paired_out"] + ".txt", [in_header, spk_header, tgt_header])
data_proc.build_vocab_files()
data_proc.write_to_tfrecord(args["paired_out"] + ".tfrecord")
in_header.vocab_mode = "read"
tgt_header.vocab_mode = "read"

data_proc = DataProcessor(args["unpaired_out"] + ".txt", [spk_header, in_header])
data_proc.write_to_tfrecord(args["unpaired_out"] + ".tfrecord")

data_proc = DataProcessor(args["paired_validation_out"] + ".txt", [in_header, spk_header, tgt_header])
data_proc.write_to_tfrecord(args["paired_validation_out"] + ".tfrecord")
