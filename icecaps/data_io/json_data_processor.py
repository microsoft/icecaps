import sys
import tensorflow as tf
import json

from icecaps.data_io.data_header import DataHeader
from icecaps.data_io.abstract_data_processor import AbstractDataProcessor
from icecaps.util.trees import TreeNode


class JSONDataProcessor(AbstractDataProcessor):
    @staticmethod
    def length_limit(line):
        if isinstance(line, TreeNode):
            if line.size() > limit:
                return None
        return super().length_limit(line)

    @staticmethod
    def lcrs_transform(line):
        if isinstance(line, TreeNode):
            return line.left_child_right_sibling()
        return line

    @staticmethod
    def fill_transform(line):
        if isinstance(line, TreeNode):
            return line.fill()
        return line

    def parse_json_obj(self, obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            root = TreeNode(obj)
            unresolved = [root]
            while unresolved:
                curr_node = unresolved[0]
                if isinstance(curr_node.value, list):
                    for i in range(1, len(curr_node.value)):
                        unresolved.append(TreeNode(curr_node.value[i], curr_node))
                    curr_node.value = curr_node.value[0]
                del unresolved[0]
            return root

    def row_gen_single_file(self, json_fname, line_shard_len=None, chunk_size=65536):
        with open(json_fname, 'r') as json_f:
            def json_str_gen():
                num_left = 0
                num_right = 0
                str_ = ""
                chunk = json_f.read(chunk_size)
                while chunk:
                    start_idx = 0
                    for i in range(len(chunk)):
                        if chunk[i] == '{':
                            if num_left == 0:
                                start_idx = i
                            num_left += 1
                        elif chunk[i] == '}':
                            num_right += 1
                            if num_left == num_right:
                                end_idx = i + 1
                                str_ += chunk[start_idx:end_idx]
                                yield json.loads(str_)
                                str_ = ""
                                num_left = 0
                                num_right = 0
                    if num_left > 0:
                        str_ += chunk[start_idx:]
                    chunk = json_f.read(chunk_size)
            data = json_str_gen()
            for obj in data:
                field_map = {field: self.parse_json_obj(obj[field]) for field in obj}
                field_ls = [field_map[self.headers[i].name] for i in range(len(self.headers))]
                yield field_ls

    def concatenate_segments(self, row):
        return {self.headers[i].name : row[i] for i in range(len(row))}

    def jsonify(self, obj):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, TreeNode):
            if obj.is_leaf():
                return obj.value
            values = [obj.value]
            unresolved = [(obj, values)]
            while unresolved:
                curr_node, curr_node_ls = unresolved[0]
                for child in curr_node.children:
                    child_ls = [child.value]
                    curr_node_ls.append(child_ls)
                    unresolved.append((child, child_ls))
                del unresolved[0]
            return values

    def write_tree_sets_to_json(self, json_fname, tree_sets):
        obj_list = [{field: jsonify(tree_set[field])
                     for field in tree_set} for tree_set in tree_sets]
        with open(json_fname, 'w') as json_f:
            json_f.write(json.dumps(obj_list))

    def append_tree_set_to_json(self, json_f, tree_set):
        obj_list = {field: jsonify(tree_set[field]) for field in tree_set}
        if json_f.tell() == 0:
            json_f.write('[' + json.dumps(obj_list))
        else:
            json_f.write(',' + json.dumps(obj_list))

    def finish_appending_to_json(self, json_f):
        json_f.write(']')

    def get_max_width(self, json_fname, line_gen=None):
        print("Getting maximum width..")
        width_map = dict()
        if line_gen is None:
            line_gen = self.row_gen()
        for row in line_gen:
            tree_set = self.concatenate_segments(row)
            for field in tree_set:
                if isinstance(tree_set[field], TreeNode):
                    if field in width_map:
                        width_map[field] = max(
                            width_map[field], tree_set[field].width())
                    else:
                        width_map[field] = tree_set[field].width()
        return width_map

    def write_to_json(self, out_file, pipeline=None, max_lines=None, line_gen=None):
        print("Writing to txt..")
        with open(out_file, "w", encoding="utf8") as out_f:
            line_ctr = 0
            if line_gen is None:
                line_gen = self.row_gen()
            for row in line_gen:
                if not self.process_row(row, pipeline):
                    continue
                tree_set = self.concatenate_segments(row)
                append_tree_set_to_json(out_f, tree_set)
                line_ctr = self.print_lines_processed(line_ctr, "trees")
                if max_lines is not None and line_ctr >= max_lines:
                    break
            finish_appending_to_json(out_f)
        self.in_files = [out_file]
        return out_file

