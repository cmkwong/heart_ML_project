import numpy as np
import collections
import random

output = collections.namedtuple('output', field_names=['conditions', 'data', 'target'])
output.conditions = []
output.data = []
output.target = []

class TF_condition:
    def __init__(self, feature, c_index):
        self.feature = feature
        self.c_index = c_index
        self.description = str(feature)

    def judge(self, x):
        if x[self.c_index]:
            return True
        elif not x[self.c_index]:
            return False

    def cell_split(self, data_arr, target_col):
        data_col = data_arr[:, self.c_index]
        data_col.reshape(target_col.shape[0], )
        target_col.reshape(target_col.shape[0], )
        # split the cell
        l_index = data_col == True
        r_index = data_col == False
        left_cell = target_col[l_index]
        rigth_cell = target_col[r_index]
        return left_cell, rigth_cell, l_index, r_index

class numeric_condition:
    def __init__(self, feature, c_index, k_value):
        self.feature = feature
        self.c_index = c_index
        self.k_value = k_value
        self.description = str(feature) + " < " + str(k_value)

    def judge(self, x):
        if x[self.c_index] <= self.k_value:
            return True
        elif x[self.c_index] > self.k_value:
            return False

    def cell_split(self, data_arr, target_col):
        data_col = data_arr[:, self.c_index]
        data_col.reshape(target_col.shape[0], )
        target_col.reshape(target_col.shape[0], )
        # split the cell
        l_index = data_col <= self.k_value
        r_index = data_col > self.k_value
        left_cell = target_col[l_index]
        rigth_cell = target_col[r_index]
        return left_cell, rigth_cell, l_index, r_index

class MC_condition:
    def __init__(self, feature, c_index, k_values):
        self.feature = feature
        self.c_index = c_index
        self.k_values = k_values
        self.description = str(feature) + " in " + str(k_values)

    def judge(self, x):
        if x[self.c_index] in self.k_values:
            return True
        else:
            return False

    def cell_split(self, data_arr, target_col):
        data_col = data_arr[:, self.c_index]
        data_col.reshape(target_col.shape[0], )
        target_col.reshape(target_col.shape[0], )
        # split the cell
        l_index, r_index = [], []
        for d in data_col:
            if d in self.k_values:
                l_index.append(True)
                r_index.append(False)
            else:
                l_index.append(False)
                r_index.append(True)
        l_index, r_index = np.array(l_index), np.array(r_index)
        left_cell = target_col[l_index]
        rigth_cell = target_col[r_index]
        return left_cell, rigth_cell, l_index, r_index

class rank_condition:
    def __init__(self, feature, c_index, k_value):
        self.feature = feature
        self.c_index = c_index
        self.k_value = k_value
        self.description = str(feature) + " < " + k_value

    def judge(self, x):
        if x[self.c_index] <= self.k_value:
            return True
        elif x[self.c_index] > self.k_value:
            return False

    def cell_split(self, data_arr, target_col):
        data_col = data_arr[:, self.c_index]
        data_col.reshape(target_col.shape[0], )
        target_col.reshape(target_col.shape[0], )
        # split the cell
        l_index = data_col <= self.k_value
        r_index = data_col > self.k_value
        left_cell = target_col[l_index]
        rigth_cell = target_col[r_index]
        return left_cell, rigth_cell, l_index, r_index

class DecisionTree:
    def __init__(self, tolerance, max_layer=100):
        self.tolerance = tolerance
        self.max_layer = max_layer
        self.num_layer = 0

    def drop_variables(self, conditions, total_variables_num):
        raise NotImplementedError("override drop_variables")

    def drop_variables_p(self, conditions):
        raise NotImplementedError("override drop_variables")

    def info_gain(self, parent_entropy, children_entropy):
        return parent_entropy - children_entropy

    def get_set(self, arr_col):
        arr_col.reshape(arr_col.shape[0],)
        element_set = set()
        for a in arr_col:
            element_set.add(a)
        return element_set

    def num2label(self, arr_col):
        arr_col.reshape(arr_col.shape[0], )
        label_arr_col = np.empty(arr_col.shape, dtype=str)
        labels_sample = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        element_set = self.get_set(arr_col)
        for i, element in enumerate(element_set):
            label = labels_sample[i]
            for ii, value in enumerate(arr_col):
                if value == element:
                    label_arr_col[ii] = label
        return label_arr_col

    def calc_cell_entropy(self, arr_col):
        arr_col.reshape(arr_col.shape[0],)
        element_set = self.get_set(arr_col)
        # init dict
        element_statistic = {}
        for element in element_set:
          element_statistic[element] = 0
        # scan each row
        total = 0
        for a in arr_col:
            for element in element_set:
                if a == element:
                    element_statistic[element] = element_statistic[element] + 1
                    total = total + 1

        # return entropy
        entropy = 0
        for _, value in element_statistic.items():
            p = value / total
            entropy = entropy + (-1) * p * np.log(p)
        return entropy

    def calc_childs_entropy(self, arr_list):
        # layer total element
        total = 0
        for arr_col in arr_list:
            total = total + arr_col.shape[0]
        # calc childs entropy
        childs_entropy = 0
        for arr_col in arr_list:
            cell_entropy = self.calc_cell_entropy(arr_col)
            weight = arr_col.shape[0] / total
            childs_entropy = childs_entropy + weight * cell_entropy
        return childs_entropy

    def num_split_pos(self, arr_col):
        cut_pos = []
        element_arr = np.sort(np.array(list(self.get_set(arr_col))))
        if len(element_arr) == 1:
            cut_pos.append(element_arr[0])
        elif len(element_arr) > 1:
            for i in np.arange(0, len(element_arr)-1):
                cut_pos.append(np.mean([element_arr[i], element_arr[i+1]]))
        return cut_pos

    def mc_combination(self, arr_col):
        combinations = []
        element_arr = np.array(list(self.get_set(arr_col)))
        for i in np.arange(0, len(element_arr)-1):
            combinations.append([element_arr[i]])
            for ii in np.arange(i+1, len(element_arr)):
                combination = []
                combination.append(element_arr[i])
                combination.append(element_arr[ii])
                combinations.append(combination)
        combinations.append([element_arr[-1]])
        return combinations

    def rk_split_pos(self, arr_col):
        rank = []
        element_arr = np.array(list(self.get_set(arr_col)))
        for r in np.arange(0, np.max(element_arr)):
            rank.append(r)
        return rank

    def condition_build(self, data_array, feature_list, type_list):
        conditions = []
        for c_index in np.arange(data_array.shape[1]):
            data_col = data_array[:,c_index]
            if (type_list[c_index] == "tf"):
                conditions.append(TF_condition(feature_list[c_index], c_index))
            elif (type_list[c_index] == "num"):
                positions = self.num_split_pos(data_col)
                for position in positions:
                    conditions.append(numeric_condition(feature_list[c_index], c_index,position))
            elif (type_list[c_index] == "mc"):
                combinations = self.mc_combination(data_col)
                for combination in combinations:
                    conditions.append(MC_condition(feature_list[c_index], c_index,combination))
            elif (type_list[c_index] == "rk"):
                ranks = self.rk_split_pos(data_col)
                for rank in ranks:
                    conditions.append(rank_condition(feature_list[c_index], c_index,rank))
        return conditions

    def cell_satisfied(self, arr_col):
        arr_col.reshape(arr_col.shape[0], )
        element_set = self.get_set(arr_col)
        total = arr_col.shape[0]
        # percentages = []
        labels = []
        for label in element_set:
            percentage = np.sum(arr_col == label) / total
            if percentage >= (1-self.tolerance):
                # percentages.append(np.sum(arr_col == label) / total)
                labels.append(label)
        if len(labels) > 0:
            # in case there has several dominant labels
            index = random.randint(0, len(labels)-1)
            return "done", labels[index]
        return "notdone", None

    def dominant_label(self, arr_col):
        arr_col.reshape(arr_col.shape[0], )
        element_set = self.get_set(arr_col)
        total = arr_col.shape[0]
        labels = []
        max_percentage = 0
        for label in element_set:
            percentage = np.sum(arr_col == label) / total
            if percentage >= max_percentage:
                max_percentage = percentage
                labels.append(label)
        # in case there has several dominant labels
        index = random.randint(0, len(labels)-1)
        return labels[index]

    def overall_cells_split(self, conditions, data_arr, target_col):
        child_cells = collections.namedtuple("child_cells", field_names=["l_data", "r_data", "l_cell", "r_cell", "condition"])
        target_col.reshape(target_col.shape[0], )
        # calc parent entropy
        parent_entropy = self.calc_cell_entropy(target_col)
        # init params
        best_info_gain = 0
        best_condition = None
        best_left_cell, best_right_cell, best_l_index, best_r_index = None, None, None, None
        for i, condition in enumerate(conditions):
            left_cell, right_cell, l_index, r_index = condition.cell_split(data_arr, target_col)
            childs_entropy = self.calc_childs_entropy([left_cell, right_cell])
            info_gain = self.info_gain(parent_entropy, childs_entropy)
            # the best info gain condition
            if info_gain > best_info_gain:
                best_left_cell, best_right_cell, best_l_index, best_r_index = left_cell, right_cell, l_index, r_index
                best_info_gain = info_gain
                best_condition = condition

        # assign value
        child_cells.l_data = data_arr[best_l_index,:]
        child_cells.r_data = data_arr[best_r_index,:]
        child_cells.l_cell = best_left_cell
        child_cells.r_cell = best_right_cell
        child_cells.condition = best_condition
        return child_cells

    def build_tree(self, conditions, data_arr, target_col, drop_variable_each_step=False):

        self.num_layer = self.num_layer + 1
        show_tree, prod_tree = None, None
        status, label = self.cell_satisfied(target_col)

        # when reach the max number of layer, find the dominant label
        if self.num_layer > self.max_layer:
            label = self.dominant_label(target_col)
            status = "done"

        if status == "notdone":
            # split the cell
            if drop_variable_each_step:
                # conditions = self.drop_variables(conditions, data_arr.shape[1])
                conditions = self.drop_variables_p(conditions)
            child_cells = self.overall_cells_split(conditions, data_arr, target_col)

            # init_tree
            show_tree = {child_cells.condition.description: {}}
            prod_tree = {child_cells.condition: {}}

            show_tree[child_cells.condition.description]["True"], prod_tree[child_cells.condition][True] = self.build_tree(conditions, child_cells.l_data, child_cells.l_cell)
            show_tree[child_cells.condition.description]["False"], prod_tree[child_cells.condition][False] = self.build_tree(conditions, child_cells.r_data, child_cells.r_cell)
        elif status == "done":
            self.num_layer = 0
            return label, label
        return show_tree, prod_tree

    def predict(self, row_data_arr, prod_tree):
        row_data_arr.reshape(row_data_arr.shape[0], )
        while(True):
            condition = list(prod_tree.keys())[0]
            selection = condition.judge(row_data_arr)
            prod_tree = prod_tree[condition][selection]
            if type(prod_tree) is not dict:
                predicted_label = prod_tree  # it is label, if it is not a dict
                return predicted_label

    def calc_acc(self, predicted, target):
        predicted.reshape(predicted.shape[0], )
        target.reshape(target.shape[0], )
        total = predicted.shape[0]
        return (np.sum(predicted == target) / total) * 100

class RandomForest(DecisionTree):
    def __init__(self, tolerance, num_col_blocked_each_step, keep_probability):
        super(RandomForest, self).__init__(tolerance)
        self.num_col_blocked_each_step = num_col_blocked_each_step
        self.keep_probability = keep_probability

    def bootstrap_data(self, data_arr, OOB_percentage):
        data_length = data_arr.shape[0]
        OOB_total = int(data_length * OOB_percentage)
        # draw OOB out of the data array
        OOB_index = random.sample(range(0, data_length), OOB_total)

        # build the data arr index
        bootstrap_index = []
        for i in np.arange(data_length):
            if not (i in OOB_index):
                bootstrap_index.append(i)
        for _ in np.arange(data_length-len(bootstrap_index)):
            bootstrap_index.append(random.sample(bootstrap_index , 1)[0])

        # get the bootstrap data arr
        bootstrap_arr = None
        for c, i in enumerate(bootstrap_index):
            row_arr = data_arr[i,:].reshape(1,-1)
            if c == 0:
                bootstrap_arr = row_arr
            else:
                bootstrap_arr = np.append(bootstrap_arr, row_arr, axis=0)
        return bootstrap_arr, OOB_index

    def drop_variables(self, conditions, total_variables_num):
        col_blocked = random.sample(range(0,total_variables_num), self.num_col_blocked_each_step)
        conditions_available = []
        for condition in conditions:
            if not (condition.c_index in col_blocked):
                conditions_available.append(condition)
        return conditions_available

    def drop_variables_p(self, conditions):
        condition_keep = random.sample( range(0, len(conditions)),
                                           int(len(conditions)*self.keep_probability) )
        conditions_available = []
        for index in np.arange(len(conditions)):
            if index in condition_keep:
                conditions_available.append(conditions[index])
        return conditions_available

    def OOB_tree(self, row_index, trees, OOB_indexs):
        num_trees = len(trees)
        trees_available = []
        for i in np.arange(num_trees):
            if row_index in OOB_indexs[i]:
                trees_available.append(trees[i])
        return trees_available

    def OOB_error(self, data_arr, target_col, trees, OOB_indexs):
        predicted_labels = []
        for index in np.arange(data_arr.shape[0]):
            oob_trees = self.OOB_tree(index, trees, OOB_indexs)
            if len(oob_trees) == 0:
                print("No OOB tree available")
                target_col = np.delete(target_col, index, 0)
                continue
            oob_labels = []
            for oob_tree in oob_trees:
                oob_labels.append(self.predict(data_arr[index,:], oob_tree))
            predicted_labels.append(self.dominant_label(np.array(oob_labels)))
        return (100 - self.calc_acc(np.array(predicted_labels), target_col))




