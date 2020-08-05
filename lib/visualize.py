import numpy as np

class Drawer:
    def __init__(self, digraph, target_type):
        self.dot = digraph
        self.target_type = target_type

    def draw(self, parent_key, children_key, label):
        self.dot.edge(parent_key, children_key, label)

    def get_left_key(self, parent):
        p_key = self.get_parent_key(parent)
        label = "True"
        return list(parent[p_key]["True"].keys())[0], label

    def get_right_key(self, parent):
        p_key = self.get_parent_key(parent)
        label = "False"
        return list(parent[p_key]["False"].keys())[0], label

    def get_left_dict(self, parent):
        p_key = self.get_parent_key(parent)
        label = "True"
        return parent[p_key]["True"], label

    def get_right_dict(self, parent):
        p_key = self.get_parent_key(parent)
        label = "False"
        return parent[p_key]["False"], label

    def get_parent_key(self, parent):
        p_key = list(parent.keys())[0]
        return p_key

    def interpret(self, parent):

        parent_key = self.get_parent_key(parent)

        # left
        left_dict, l_label = self.get_left_dict(parent)
        if type(left_dict) == dict:
            children_key = self.get_parent_key(left_dict)
            self.draw(parent_key, children_key, l_label)
            self.interpret(parent[parent_key]["True"])
        elif type(left_dict) == self.target_type:
            self.draw(parent_key, left_dict, l_label)

        # right
        right_dict, r_label = self.get_right_dict(parent)
        if type(right_dict) == dict:
            children_key = self.get_parent_key(right_dict)
            self.draw(parent_key, children_key, r_label)
            self.interpret(parent[parent_key]["False"])
        elif type(right_dict) == self.target_type:
            self.draw(parent_key, right_dict, r_label)