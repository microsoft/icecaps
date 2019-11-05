import sys
import os

from icecaps.util.trees import TreeNode

class Vocabulary:
    def __init__(self, size=3, fname=None, skip_tokens='', skip_tokens_start=''):
        self.special_tokens = self.END, self.GO, self.UNK = ('_END', '_GO', '_UNK')
        if fname is not None:
            if os.path.isfile(fname):
                with open(fname, 'r', encoding="utf8") as f:
                    self.words = []
                    for i in f:
                        self.words.append(i.strip())                
            else:
                with open(fname, 'w', encoding="utf8") as f:
                    self.words = [self.END, self.GO, self.UNK]
                    for i in self.words:
                        f.write(i + "\n")
        else:
            self.words = [self.END, self.GO, self.UNK] + \
                [str(i) for i in range(3, size)]
        assert (self.END in self.words and self.UNK in self.words), "Vocab file doesn't match expected set of special tokens"
        self.word2idx = dict((c, i) for i, c in enumerate(self.words))
        self.idx2word = dict((i, c) for i, c in enumerate(self.words))
        self.word_counts = dict((c, 0) for c in self.words)
        try:
            self.start_token_id = self.word2idx[self.GO]
        except:
            pass
            #print(
             #   "== No _GO token found in vocabulary. Using soft_W(encoding) as seed instead==")
        self.end_token_id = self.word2idx[self.END]
        self.unk_token_id = self.word2idx[self.UNK]
        self.populate_skip_tokens(skip_tokens, skip_tokens_start)


    def size(self):
        return len(self.words)

    def populate_skip_tokens(self, skip_tokens, skip_tokens_start, skip_semi_first=True):
        self.skip_tokens = []
        self.skip_tokens_start = []
        if skip_tokens == '' and skip_tokens_start == '':
            return
        tokens = skip_tokens.split(';')
        for token in tokens:
            try:
                self.skip_tokens.append(self.word2idx[token])
                self.skip_tokens_start.append(self.word2idx[token])
            except:
                print(
                    "== Token %s not found in vocabulary. Skipping (for skip_tokens)==" % (token))
                continue
        tokens = skip_tokens_start.split(';')
        if skip_semi_first and ';' in self.word2idx:
            self.skip_tokens_start.append(self.word2idx[';'])
        for token in tokens:
            try:
                self.skip_tokens_start.append(self.word2idx[token])
            except:
                print(
                    "== Token %s not found in vocabulary. Skipping (for skip_tokens_start)==" % (token))
                continue
        self.skip_tokens.sort()
        self.skip_tokens_start.sort()

    def get_token_id(self, token, fixed_vocab=True):
        token = str(token)
        if token in self.word2idx:
            self.word_counts[token] += 1
            return self.word2idx[token]
        elif not fixed_vocab:
            self.word2idx[token] = len(self.words)
            self.idx2word[len(self.words)] = token
            self.words.append(token)
            self.word_counts[token] = 1
            return self.word2idx[token]
        else:
            self.word_counts[self.UNK] += 1
            return self.unk_token_id

    def tokenize(self, sentence, fixed_vocab=True, by_char=False, pad_len=None, take_first_n=False):
        if isinstance(sentence, str):
            sentence_split = list(
                sentence) if by_char else sentence.strip().split()
        elif isinstance(sentence, TreeNode):
            sentence_split = [node.value for node in sentence.depth_first_traversal()]
        else:
            raise ValueError("Invalid type.")
        if take_first_n:
            sentence_split = sentence_split[:pad_len]
        tokens = []
        for i in range(len(sentence_split)):
            tokens.append(self.get_token_id(sentence_split[i], fixed_vocab))
        if pad_len is not None:
            assert(len(tokens) <= pad_len)
            tokens = tokens + [self.end_token_id] * (pad_len - len(tokens))
        return tokens

    def count_cutoff(self, cutoff=3):
        new_words = []
        for i in range(len(self.words)):
            if self.words[i] in ('_END', '_GO', '_UNK') or self.word_counts[self.words[i]] >= cutoff:
                new_words.append(self.words[i])
        self.words = new_words
        self.word2idx = dict((c, i) for i, c in enumerate(self.words))
        self.idx2word = dict((i, c) for i, c in enumerate(self.words))
        new_word_counts = dict()
        for word in self.word_counts:
            if word in self.word2idx:
                new_word_counts[word] = self.word_counts[word]
        self.word_counts = new_word_counts
