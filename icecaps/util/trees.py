import sys
import numpy as np


class TreeNode:
    def __init__(self, value, parent=None, children=None, mask=None):
        self.value = value
        self.parent = parent
        self.children = children if children else []
        self.mask = mask if mask is not None else [1.0] * 81
        self._positional_encoding = None
        if self.parent is None:
            self.branch = 0
        else:
            if self not in self.parent.children:
                self.parent.children.append(self)
            self.branch = self.parent.children.index(self)

    def num_children(self):
        return len(self.children)

    def size(self):
        return 1 + sum([child.size() for child in self.children])

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max([child.depth() for child in self.children])

    def height(self):
        if self.parent is None:
            return 0
        return 1 + self.parent.height()

    def width(self):
        return max([self.num_children()] + [child.width() for child in self.children])

    def is_leaf(self):
        return self.num_children() == 0

    def is_first_child(self):
        return self.branch == 0

    def is_last_child(self):
        return self.branch == self.parent.num_children() - 1 if self.parent else True

    def get_positional_encoding(self):
        if self._positional_encoding is None:
            if self.parent:
                self._positional_encoding = [
                    0.0 for _ in range(self.parent.num_children())]
                self._positional_encoding[self.branch] = 1.0
                self._positional_encoding += self.parent.get_positional_encoding()
            else:
                self._positional_encoding = []
        return self._positional_encoding

    def get_padded_positional_encoding(self, max_pos_len):
        padded = [x for x in self.get_positional_encoding()]
        while len(padded) < max_pos_len:
            padded.append(0.0)
        padded = padded[: max_pos_len]
        return padded

    def is_isomorphic(self, arg, struct_only=False):
        if (struct_only or self.value == arg.value) and self.num_children() == arg.num_children():
            for i in range(len(self.children)):
                if not self.children[i].is_isomorphic(arg.children[i], struct_only):
                    return False
            return True
        return False

    def prefix_traversal(self):
        def _prefix(node):
            yield node
            for child in node.children:
                yield from _prefix(child)
        yield from _prefix(self)

    def postfix_traversal(self):
        def _postfix(node):
            for child in node.children:
                yield from _postfix(child)
            yield node
        yield from _postfix(self)

    def depth_first_traversal(self):
        yield from self.prefix_traversal()

    def breadth_first_traversal(self):
        unresolved = [self]
        while unresolved:
            yield unresolved[0]
            unresolved += unresolved[0].children
            del unresolved[0]

    def choose_traversal(self, str_):
        str_to_traversal = {
            "prefix": self.prefix_traversal,
            "postfix": self.postfix_traversal,
            "depth_first": self.depth_first_traversal,
            "breadth_first": self.breadth_first_traversal
        }
        yield from str_to_traversal[str_]()

    def convert_to_sequence(self, traversal, separator=' '):
        seq = ""
        for node in traversal:
            seq += str(node.value) + separator
        return seq

    def fill(self, branch_factor=2, placeholder_token='_NULL'):
        fill_tree = {}
        for node in self.depth_first_traversal():
            value = node.value
            if node.is_leaf():
                value += "_0"
            if node is self:
                fill_tree[node] = TreeNode(value)
            else:
                fill_tree[node] = TreeNode(value, fill_tree[node.parent])
        for node in self.depth_first_traversal():
            if not node.is_leaf():
                while len(fill_tree[node].children) < branch_factor:
                    TreeNode(placeholder_token, fill_tree[node])
        return fill_tree[self]

    def left_child_right_sibling(self, placeholder_token='_NULL'):
        lcrs_tree = {}
        for node in self.depth_first_traversal():
            if node is self:
                lcrs_tree[node] = TreeNode(node.value)
            else:
                if node.is_first_child():
                    lcrs_tree[node] = TreeNode(
                        node.value, lcrs_tree[node.parent])
                    if node.parent.is_last_child():
                        TreeNode(placeholder_token, lcrs_tree[node.parent])
                else:
                    lcrs_tree[node] = TreeNode(
                        node.value, lcrs_tree[node.parent.children[node.branch - 1]])
                if node.is_leaf():
                    TreeNode(placeholder_token, lcrs_tree[node])
                    if node.is_last_child():
                        TreeNode(placeholder_token, lcrs_tree[node])
        return lcrs_tree[self]

    def inverse_left_child_right_sibling(self, placeholder_token='_NULL'):
        ilcrs_tree = {}
        try:
            for node in self.depth_first_traversal():
                if node.num_children() == 1:
                    TreeNode(placeholder_token, node)
            for node in self.depth_first_traversal():
                if node is self:
                    ilcrs_tree[node] = TreeNode(node.value)
                elif node.value != placeholder_token:
                    true_first_child = node
                    while true_first_child.branch == 1:
                        true_first_child = true_first_child.parent
                    ilcrs_tree[node] = TreeNode(
                        node.value, ilcrs_tree[true_first_child.parent])
            return ilcrs_tree[self]
        except:
            return TreeNode(placeholder_token)


def build_tree_from_traversal(values, traversal="dfs", branch_factor=2, temp=None):
    if traversal == "dfs":
        curr_node = None
        root = TreeNode("_NULL")
        for value in values:
            if value == "_END":
                continue
            if temp and value[0] == '*':
                value = temp + value
            curr_node = TreeNode(value, curr_node)
            if curr_node.parent is None:
                root = curr_node
            if curr_node.value[-2:] == "_0" or curr_node.value == '_NULL':
                while curr_node.parent is not None and (curr_node.num_children() == 0 or curr_node.num_children() == branch_factor):
                    curr_node = curr_node.parent
            elif curr_node.value[-2:] == "_1":
                TreeNode('_NULL', curr_node)
        return root
    if traversal == "bfs":
        nodes = []
        curr_parent = None
        for value in values:
            curr_node = TreeNode(value, curr_parent)
            if len(nodes) == 0:
                root = curr_node
            if value[-2:] == '_1':
                TreeNode('_NULL', curr_node)
            nodes.append(curr_node)
            if curr_parent is None:
                curr_parent = nodes[0]
            while len(nodes) > 0 and (curr_parent.num_children() == branch_factor or curr_parent.value[-2:] == '_0' or curr_parent.value == '_NULL'):
                del nodes[0]
                if len(nodes) > 0:
                    curr_parent = nodes[0]
        return root


def get_degrees(vocab, placeholder_token='_NULL'):
    # From each token, extract the number of leaves it requires.
    degrees = []
    for i in range(len(vocab.idx2word)):
        if vocab.idx2word[i].endswith("_0") or vocab.idx2word[i] == placeholder_token or vocab.idx2word[i] == vocab.end_token_id:
            degrees.append(0)
        elif vocab.idx2word[i].endswith("_1"):
            degrees.append(1)
        else:
            degrees.append(2)
    return degrees


def layer_score(tree_pairs, layers, use_ilcrs=False):
    num_correct = 0
    num_total = 0
    for src, tgt in tree_pairs:
        num_total += 1
        if use_ilcrs:
            srci = src.inverse_left_child_right_sibling()
            tgti = tgt.inverse_left_child_right_sibling()
        else:
            srci = src
            tgti = tgt
        signal = srci.size() == tgti.size()
        try:
            for srcn, tgtn in zip(srci.breadth_first_traversal(), tgti.breadth_first_traversal()):
                if srcn.height() > layers and tgtn.height() > layers:
                    signal = True
                    break
                if srcn.height() != tgtn.height() or srcn.value != tgtn.value:
                    signal = False
                    break
        except Exception:
            signal = False
        if signal:
            num_correct += 1
        print("Score " + str(layers) + ": " + str(signal))
    return num_correct / num_total


def f1_score(tree_pairs, use_ilcrs=False):
    num_total = 0
    total_f1 = 0.0
    for src, tgt in tree_pairs:
        num_total += 1
        if use_ilcrs:
            srci = src.inverse_left_child_right_sibling()
            tgti = tgt.inverse_left_child_right_sibling()
        else:
            srci = src
            tgti = tgt
        srcset = set()
        tgtset = set()
        for srcn in srci.depth_first_traversal():
            if not srcn.is_leaf():
                production = [srcn.value]
                for child in srcn.children:
                    production.append(child.value)
                production = tuple(production)
                srcset.add(production)
        for tgtn in tgti.depth_first_traversal():
            if not tgtn.is_leaf():
                production = [tgtn.value]
                for child in tgtn.children:
                    production.append(child.value)
                production = tuple(production)
                tgtset.add(production)
        intersect = tgtset.intersection(srcset)
        if len(srcset) == 0 or len(tgtset) == 0 or len(intersect) == 0:
            f1 = 0.0
        else:
            precision = len(tgtset.intersection(srcset)) / len(srcset)
            recall = len(tgtset.intersection(srcset)) / len(tgtset)
            f1 = 2 * precision * recall / (precision + recall)
        total_f1 += f1
        print("Score F1: " + str(f1))
    return total_f1 / num_total
