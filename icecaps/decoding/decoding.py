import tensorflow as tf
import numpy as np
import time
import sys
import re
import itertools
from collections import defaultdict

from icecaps.io.data_processing import DataProcessor
import icecaps.util.util as util


class ScoredResult:
    def __init__(self, query, hypotheses, decode_time=-1.0, mmi_time=-1.0, speaker_id=-1):
        self.query = query
        self.hypotheses = hypotheses
        self.decode_time = decode_time
        self.mmi_time = mmi_time
        self.speaker_id = speaker_id

    def rep_penalty(self, hypothesis, repetition_allowance):
        unique_words = float(len(set(hypothesis)))
        return np.log(np.minimum(1.0, unique_words / np.ceil(repetition_allowance * len(hypothesis))))

    def computeTotalScore(self, lambda_val=0.5, repetition_penalty=0, repetition_allowance=0, length_penalty_weight=0):
        for i in range(len(self.hypotheses)):
            hypothesis_words = self.hypotheses[i].response.split(' ')
            self.hypotheses[i].total_score = (
                1.0 - lambda_val) * self.hypotheses[i].decode_score + lambda_val * self.hypotheses[i].mmi_score
            self.hypotheses[i].total_score += length_penalty_weight * \
                len(hypothesis_words)  # Add length penalty
            self.hypotheses[i].total_score += repetition_penalty * self.rep_penalty(
                hypothesis_words, repetition_allowance)  # Add repetition penalty

    def sort(self):
        self.hypotheses.sort(key=lambda x: x.total_score, reverse=True)

    @property
    def best(self):
        return self.hypotheses[0].response

    def printScoredHypotheses(self):
        print("Query " + self.query + " Speaker " + str(self.speaker_id) +
              " MMI Time " + str(self.mmi_time) + " Decode Time " + str(self.decode_time))
        for hyp in self.hypotheses:
            print(hyp.response, hyp.decode_score,
                  hyp.mmi_score, hyp.total_score)
        print("Best Response: ", self.best)


class Hypotheses:
    def __init__(self, response, decode_score, mmi_score):
        self.response = response
        self.decode_score = decode_score
        self.mmi_score = mmi_score
        self.total_score = 0.0


def convertDecodedResultToScoredResult(decoded_result, persona=False):
    query = decoded_result.query
    decode_time = decoded_result.evaluation_time
    hypotheses = []
    for hypothesis in decoded_result.hypotheses:
        hypotheses.append(Hypotheses(hypothesis.response, hypothesis.score))
    if persona:
        scored_result = ScoredResult(
            query, hypotheses, decode_time, speaker_id=decoded_result.speaker_id)
    else:
        scored_result = ScoredResult(query, hypotheses, decode_time)
    return scored_result


def populateScoredResult(scored_result, eval_scores):
    for i in range(len(eval_scores)):
        scored_result.hypotheses[i].mmi_score = eval_scores[i]


# Simple decode helper function: returns dict(inp1: [(hyp1,score1), (hyp2,score2),...], ..)
def decode(model, input_fn, vocab, outstream=sys.stdout, hooks=[], save_all=False, print_query=False):
    class NoStream():
        def write(self, _):
            pass
    if outstream is None:
        outstream = NoStream()
    predictions = model.predict(input_fn, hooks=hooks)
    inputwise_hypotheses = {}
    inputwise_hypotheses = defaultdict(lambda: [], inputwise_hypotheses)

    def hypothesis_make_string(hypothesis, spaces=True):
        eng = ""
        for k in range(len(hypothesis)):
            if hypothesis[k] < 0 or hypothesis[k] == vocab.end_token_id:
                break
            eng += vocab.idx2word[hypothesis[k]]
            if spaces:
                eng += ' '
        return eng

    current_query = ""
    current_response = ""
    current_speaker = np.ones(1) * -1
    i = 0
    for pred in predictions:
        i += 1
        processed_inp = ''
        if current_query != hypothesis_make_string(pred["inputs"]) or ("speaker_ids" in pred and not np.array_equal(current_speaker, pred["speaker_ids"])):
            inputwise_hypotheses = {}
            inputwise_hypotheses = defaultdict(lambda: [], inputwise_hypotheses)
            current_query = hypothesis_make_string(pred["inputs"])
            if "speaker_ids" in pred:
                current_speaker = pred["speaker_ids"]
            if print_query:
                if current_query != "":
                    outstream.write("\n")
                outstream.write("Inputs: " + current_query + "\n")
            inp = itertools.takewhile(
                lambda x: x != vocab.end_token_id, pred["inputs"])  # strip the EOS
            processed_inp = ' '.join(vocab.idx2word[idx] for idx in list(inp))
            if "targets" in pred:
                outstream.write(
                    "Targets: " + hypothesis_make_string(pred["targets"]) + "\n")
            if not save_all and len(hooks) > 0 and isinstance(hooks[0], InteractiveInputHook):
                hooks[0].context_ls.append(current_query.split(hooks[0].eos_token + ' ')[-1])
                hooks[0].context_ls.append(hypothesis_make_string(pred["outputs"]))
                hooks[0].context_ls = hooks[0].context_ls[2:]
        elif current_response == hypothesis_make_string(pred["outputs"]):
            continue
        current_response = hypothesis_make_string(pred["outputs"])
        inputwise_hypotheses[processed_inp].append(pred)
        #(pred["inputs"], pred["scores"], pred["outputs"], pred["speaker_ids"]))
        if save_all and len(hooks) > 0 and isinstance(hooks[0], InteractiveInputHook):
            hooks[0].pre_mmi = inputwise_hypotheses[processed_inp]
        outstream.write("%f : %s\n" % (
            pred["scores"], hypothesis_make_string(pred["outputs"])))
    for _, v in inputwise_hypotheses.items():
        v.sort(key=lambda x: x["scores"], reverse=True)
    return inputwise_hypotheses


def serving_input_fn(in_len, tgt_len):
    def true_serving_input_fn():
        # Consider making batch size None to accept any batch size
        inputs = tf.placeholder(
            tf.int64, shape=[None, in_len])
        targets = tf.placeholder(
            tf.int64, shape=[None, tgt_len])
        speaker_id = tf.placeholder(tf.int64, shape=[None, 1])
        forward_scores = tf.placeholder(tf.float32, shape=[None, 1])
        features = {'inputs': inputs, 'targets': targets,
                    'speaker_ids': speaker_id, 'forward_scores': forward_scores}
        return tf.estimator.export.ServingInputReceiver(features, features)
    return true_serving_input_fn



class InteractiveInputHook(tf.train.SessionRunHook):
    _step = 0
    _session = None
    last_input_time = None

    def __init__(self, input_fn, num_turns=1, eos_token='', mmi_component=None, lambda_balance=None, vocab=None):
        self._input_fn = input_fn
        self.num_turns = num_turns
        self.eos_token = eos_token
        self.mmi_component = mmi_component
        self.lambda_balance = lambda_balance if lambda_balance is not None else 0.5
        self.vocab = vocab
        self.context_ls = ['' for _ in range(num_turns - 1)]
        self.pre_mmi = None

    def set_iterators_to_init(self, iterators):
        self._iterators = iterators if isinstance(
            iterators, list) else [iterators]

    def set_iterators_enqueue(self, enqueueOps):
        self._enqueueOps = iterators if isinstance(
            enqueueOps, list) else [enqueueOps]

    def set_input_placeholder(self, input_placeholder):
        self._input_placeholder = input_placeholder

    def begin(self):
        # can still add things to the graph
        self._initializers = [iter.initializer for iter in self._iterators]

    def hypothesis_make_string(self, hypothesis, spaces=True):
        eng = ""
        for k in range(len(hypothesis)):
            if hypothesis[k] < 0 or hypothesis[k] == self.vocab.end_token_id:
                break
            eng += self.vocab.idx2word[hypothesis[k]]
            if spaces:
                eng += ' '
        return eng

    def _get_feed_dict(self):
        if self.pre_mmi is not None and len(self.pre_mmi) > 0 and self.mmi_component is not None:
            mmi_features = {
                "targets": np.asarray([x["inputs"] for x in self.pre_mmi]),
                "inputs": np.asarray([x["outputs"] for x in self.pre_mmi]),
            }
            if "speaker_ids" in self.pre_mmi[0]:
                mmi_features["speaker_ids"] = np.asarray([x["speaker_ids"] for x in self.pre_mmi])
            mmi_scores = []
            predictor = tf.contrib.predictor.from_estimator(
                self.mmi_component, serving_input_fn(len(mmi_features["inputs"][0]), len(mmi_features["targets"][0])))
            mmi_scores = predictor(mmi_features)["scores"]
            rescored = []
            for i in range(len(mmi_scores)):
                rescored.append([self.pre_mmi[i]["inputs"], (1 - self.lambda_balance) * self.pre_mmi[i]["scores"] + self.lambda_balance * mmi_scores[i], self.pre_mmi[i]["outputs"]])
            rescored.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(5, len(rescored))):
                print("%f : %s" % (
                    rescored[i][1], self.hypothesis_make_string(rescored[i][2])))
            self.context_ls.append(self.hypothesis_make_string(rescored[0][0]).split(self.eos_token + ' ')[-1])
            self.context_ls.append(self.hypothesis_make_string(rescored[0][2]))
            self.context_ls = self.context_ls[2:]
            self.pre_mmi = None

        feed_dict = {}

        context = ''
        for i in range(self.num_turns - 1):
            context += self.context_ls[i] + self.eos_token + ' '
        features = self._input_fn(self, context)

        self.last_input_time = time.time()
        if isinstance(self._input_placeholder, dict):
            assert(isinstance(features, dict))
            for k in self._input_placeholder:
                assert(k in features)
                feed_dict[self._input_placeholder[k]] = features[k]
        else:
            feed_dict = {self._input_placeholder: features}
        return feed_dict

    def before_run(self, run_context):
        self._step = self._step + 1
        if self._step > 1:
            self._session.run(self._initializers +
                              self._enqueueOps, self._get_feed_dict())
        return None

    def after_create_session(self, session, coord):
        # no more changes to the graph, can only execute certain nodes
        del coord
        session.run(self._initializers + self._enqueueOps,
                    self._get_feed_dict())
        self._session = session


def interactive_input_fn(hook, queue, placeholders, field_map):
    enqueueOp = queue.enqueue(placeholders)
    inputFeatures = queue.dequeue()
    dataset = tf.data.Dataset.from_tensors(inputFeatures)
    iterator = dataset.make_initializable_iterator()
    hook.set_iterators_to_init(iterator)
    hook.set_iterators_enqueue(enqueueOp)
    hook.set_input_placeholder(placeholders)
    next_elem = iterator.get_next()
    features = {field: next_elem[field_map[field]] for field in field_map}
    return features


def interactive_input_fn_simple(hook, max_input_len):
    msg_tokenized = tf.placeholder(tf.int64, shape=[1, max_input_len])
    queue = tf.FIFOQueue(1, dtypes=[tf.int64], shapes=[
                         (1, max_input_len)], names=['message'])
    placeholders = {'message': msg_tokenized}
    field_map = {"inputs": "message"}
    return interactive_input_fn(hook, queue, placeholders, field_map)


def interactive_input_fn_persona(hook, max_input_len):
    msg_tokenized = tf.placeholder(tf.int64, shape=[1, max_input_len])
    speaker_id = tf.placeholder(tf.int64, shape=[1, 1])
    queue = tf.FIFOQueue(1, dtypes=[tf.int64, tf.int64], shapes=[
                         (1, max_input_len), (1, 1)], names=['message', 'speaker'])
    placeholders = {'message': msg_tokenized, 'speaker': speaker_id}
    field_map = {"inputs": "message", "speaker_ids": "speaker"}
    return interactive_input_fn(hook, queue, placeholders, field_map)


def convert_interactive_input(in_msg, context, vocab, pad_len=200):
    if in_msg == "exit()":
        raise StopIteration()
    speaker = 0
    if '|' in in_msg:
        idx = in_msg.index('|')
        speaker = int(in_msg[:idx].strip())
        in_msg = in_msg[idx+1:].strip()
    in_msg = DataProcessor.basic_preprocess(in_msg)
    in_msg = context + in_msg
    tokenized_in_msg = vocab.tokenize(in_msg, pad_len=pad_len)
    return {'message': [tokenized_in_msg], 'speaker': [[speaker]]}


def interactive_decode(input_msg_listener, model, vocab, outstream=sys.stdout, mmi_component=None, persona=False, max_len=50, lambda_balance=None, num_turns=1, eos_token=''):
    def _fn_get_input(hook, context):
        print()
        while True:
            in_msg = input_msg_listener()
            if in_msg == 'reset()':
                hook.context_ls = ['' for _ in range(len(hook.context_ls))]
                context = (hook.eos_token + ' ') * len(hook.context_ls)
            else:
                break
        return convert_interactive_input(in_msg, context, vocab, pad_len=max_len)
    hook = InteractiveInputHook(_fn_get_input, num_turns, eos_token, mmi_component, lambda_balance, vocab)
    input_fn = interactive_input_fn_persona if persona else interactive_input_fn_simple
    def input_fn_src(): return input_fn(hook, max_len)
    return decode(model, input_fn_src, vocab, outstream=(None if mmi_component else outstream), hooks=[hook], save_all=(mmi_component is not None))


def cmd_decode(model, vocab, mmi_component=None, persona=False, max_len=200, lambda_balance=None, num_turns=1, eos_token=''):
    cmd_listener = lambda: input("Query: ").lower().strip()
    return interactive_decode(cmd_listener, model, vocab, sys.stdout, mmi_component, persona, max_len, lambda_balance, num_turns, eos_token)

