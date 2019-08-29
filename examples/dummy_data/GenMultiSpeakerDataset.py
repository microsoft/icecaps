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

parser = argparse.ArgumentParser(
    description='Generates dummy data with a unique persona for the autoencoder data')
parser.add_argument('-o', '--out', help='file to write the seq2seq data to',
                    default='data_simple_persona\\data_seq2seq.txt')
parser.add_argument('-ao', '--autoenc_out', help='file to write the autoencoder data to',
                    default='data_simple_persona\\data_autoenc.txt')
parser.add_argument('-avo', '--autoenc_valid_out', help='file to write the autoencoder validation data to',
                    default='data_simple_persona\\data_autoenc_valid.txt')
parser.add_argument('-co', '--corpus_out', help='file to write the autoencoder data to',
                    default='data_simple_persona\\corpus.txt')
parser.add_argument(
    '-c', '--count', help='repeat data to get minimum number of seq2seq data entries', default=10000, type=int)
parser.add_argument('-ac', '--autoenc_count',
                    help='repeat data to get minimum number of autoencoder data entries', default=2000, type=int)
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


def write_file_data(file_path, data, replicate_to_count):
    with open(file_path, 'w', encoding="utf8") as f:
        count = 0
        while(count < replicate_to_count):
            f.writelines(data)
            count += len(data)


write_file_data(args["out"], data, args["count"])


# Auto Enc Data:

autoenc_data = []
autoend_valid_data = []
persona_id += 1  # grab the next id

# add two new unique personas. Note that the vocab is a subset of the main s2s dataset. See *NOTE below.
autoenc_answers = [['a lawyer',    'a student'],
                   ['green',       'magenta'],
                   ['portland',      'london'],
                   ['25',          '20'],
                   ['700 dollars', '50 dollars'],
                   ['great',       'good']
                   ]

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

write_file_data(args["autoenc_valid_out"], autoenc_valid_data, 1)
write_file_data(args["autoenc_out"], autoenc_data, args["autoenc_count"])


def flatten_to_list(data):
    corpus = []
    for g in np.reshape(data, -1):
        if type(g) is list:
            for e in g:
                corpus.append(e + "\n")
        else:
            corpus.append(g + "\n")
    return corpus


write_file_data(args["corpus_out"], flatten_to_list(
    answer_templates) + flatten_to_list(autoenc_answers), 1)


# Create the dict and tfrecord files

cwd = os.path.dirname(os.path.abspath(__file__))
subprocess.call(["python", "../tools/create_dict.py", "--input",
                 "corpus.txt",  "--out_file", "data.dic"], cwd=cwd)
subprocess.call(["python", "../text2tfrecord.py", "--use_speaker_ids", "--vocab_file", "data.dic",
                 "--text_file", "data_seq2seq.txt", "--tfrecord_file", "data_seq2seq.tfrecord"], cwd=cwd)
subprocess.call(["python", "../text2tfrecord.py", "--use_speaker_ids", "--vocab_file", "data.dic",
                 "--text_file", "data_autoenc.txt", "--tfrecord_file", "data_autoenc.tfrecord", "--autoencode"], cwd=cwd)
subprocess.call(["python", "../text2tfrecord.py", "--use_speaker_ids", "--vocab_file", "data.dic",
                 "--text_file", "data_autoenc_valid.txt", "--tfrecord_file", "data_autoenc_valid.tfrecord"], cwd=cwd)


# *NOTE
# Model a personality that says 'Ummm', 'yeah' and 'like' a lot.
#
# This doesn't work today. We are limited to concepts and vocabulary that already exists in the
# main s2s data
# autoenc_answers=[
# [ ["yeah , I am like a {}", "yeah , I work as {}", "ummm {}"],                      'a chef'],
# [ ['ummm {} i think', 'yeah , my favorite color is {}', 'i like love {}'],          'purple'],
# [ ['yeah I am from {}', 'ummm , I live in {}', 'in like {}', 'near {}'],            'portland'],
# [ ['I am like {}', 'yeah ummm like {} years old', 'ummm about {}'],                 '24'],
# [ ['yeah , about {}', 'ummm i think around like {}', 'like {}'],                    '1234 dollars'],
# [ ["I am feeling like {}", 'ummm I am {}'],                                         'rad'],
# ]
#
# for i, question in enumerate(questions):
#     for template in autoenc_answers[i][0]:
#         entries = [ ('{}\t' + template + '\n').format(persona_id, autoenc_answers[i][1]) ]
#         autoenc_data.extend(entries)
#         entries = [(question + "\t{}\t" + template + "\n").format(persona_id, autoenc_answers[i][1])]
#         autoend_valid_data.extend(entries)
