import numpy as np
import collections

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
        if x:
            return True
        elif not x:
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
        if x <= self.k_value:
            return True
        elif x > self.k_value:
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
        if x in self.k_values:
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
        if x <= self.k_value:
            return True
        elif x > self.k_value:
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

    def info_gain(self, parent_entropy, children_entropy):
        return parent_entropy - children_entropy

    def get_set(self, arr_col):
        arr_col.reshape(arr_col.shape[0],)
        element_set = set()
        for a in arr_col:
            element_set.add(a)
        return element_set

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
        for label in element_set:
            percentage = np.sum(arr_col == label) / total
            # return if element percentage over (1-tolerance)
            if percentage >= (1-self.tolerance):
                return "done", label
        return "notdone", None

    # def tree_wrapper(self, id, left_data_arr, right_data_arr, left_cell, right_cell, condition):
    #     tree = {id: {}}
    #     tree[id]["condition"] = condition
    #     tree[id]["left"] = left_cell
    #     tree[id]["right"] = right_cell
    #     tree[id]["left_data"] = left_data_arr
    #     tree[id]["right_data"] = right_data_arr
    #     return tree

    # def cell_wrapper(self, cell_id, data_arr, target, condition):
    #     cell_dict = {cell_id: {}}
    #     cell_dict[cell_id]["condition"] = condition
    #     cell_dict[cell_id]["data"] = data_arr
    #     cell_dict[cell_id]["target"] = target

    # def merge_tree(self, mother_tree, mother_id, tree_label, direction='l'):
    #     d = None
    #     if direction == 'l':
    #         d = "left"
    #     elif direction == 'r':
    #         d = "right"
    #     mother_tree[mother_id][d] = tree_label
    #     merged_tree = mother_tree.copy()
    #     return merged_tree
    #
    # def append_node_list(self, mother_id, direction, arr_col, node_list):
    #     # LR_cell = tree[id]["left"]
    #     cell = {"id": mother_id, "direction": direction}
    #     cell_done = self.cell_satisfied(arr_col)
    #     if cell_done == False:
    #         node_list.append(cell)
    #     return node_list

    # def build_tree(self, conditions, data_arr, target_col):
    #     target_col.reshape(target_col.shape[0], )
    #     parent_entropy = self.calc_cell_entropy(target_col)
    #     best_info_gain = 0
    #     best_index = None
    #     best_condition = None
    #     left_cell, right_cell, l_index, r_index = None, None, None, None
    #     best_left_cell, best_right_cell, best_l_index, best_r_index = None, None, None, None
    #     for i, condition in enumerate(conditions):
    #         left_cell, right_cell, l_index, r_index = condition.cell_split(data_arr, target_col)
    #         childs_entropy = self.calc_childs_entropy([left_cell, right_cell])
    #         info_gain = self.info_gain(parent_entropy, childs_entropy)
    #         if info_gain > best_info_gain:
    #             best_left_cell, best_right_cell, best_l_index, best_r_index = left_cell, right_cell, l_index, r_index
    #             best_info_gain = info_gain
    #             best_index = i
    #             best_condition = condition
    #     # del conditions[best_index]
    #     tree = {best_condition: {}}
    #     l_label = self.cell_satisfied(best_left_cell)
    #     if l_label == False:
    #         new_data_arr = data_arr[best_l_index, :]
    #         new_target_col = target_col[best_l_index]
    #         tree[best_condition]["left"] = self.build_tree(conditions, new_data_arr, new_target_col)
    #     elif l_label != False:
    #         tree[best_condition]["left"] = l_label
    #         return tree
    #
    #     r_label = self.cell_satisfied(best_right_cell)
    #     if r_label == False:
    #         new_data_arr = data_arr[best_r_index, :]
    #         new_target_col = target_col[best_r_index]
    #         tree[best_condition]["right"] = self.build_tree(conditions, new_data_arr, new_target_col)
    #     elif r_label != False:
    #         tree[best_condition]["right"] = r_label
    #         return tree
    #     print("A tree build: ", tree)
    #     return tree

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

    def build_tree(self, conditions, data_arr, target_col):

        # if len(target_col) == 3:
        #     if np.sum((target_col == np.array([0,0,0]))) ==3 :
        #         print("stop")
        tree = None
        status, label = self.cell_satisfied(target_col)
        if status == "notdone":
            # split the cell
            child_cells = self.overall_cells_split(conditions, data_arr, target_col)

            # init_tree
            tree = {child_cells.condition.description: {}}

            tree[child_cells.condition.description]["True"] = self.build_tree(conditions, child_cells.l_data, child_cells.l_cell)
            tree[child_cells.condition.description]["False"] = self.build_tree(conditions, child_cells.r_data, child_cells.r_cell)
        elif status == "done":
            return label
        return tree

    # def build_tree_(self, conditions, data_arr, target_col):
    #     id = 0
    #     mother_tree = {id: {}}
    #
    #     # looping each layer
    #     for layer in np.arange(self.max_layer):
    #
    #         # looping each cell for each layer
    #         num_mother = 2 ** layer
    #         mother_ids = []
    #         child_cells = []
    #         trees = []
    #         for c in np.arange(num_mother):
    #             child_cells.append(self.overall_cells_split(conditions, data_arr, target_col))
    #             trees.append(self.tree_wrapper(id, child_cells[c].l_data, child_cells[c].r_data, child_cells[c].l_cell,
    #                                              child_cells[c].r_cell, child_cells[c].condition))
    #             mother_ids.append(id)
    #             id = id + 1
    #         for i, mother_id in enumerate(mother_ids):
    #             # mother_tree, mother_id, tree_label, direction = 'l'
    #             mother_tree = self.merge_tree(mother_tree, mother_id, trees[i], )
    #     return None
    #
    # def build_tree_2(self, conditions, data_arr, target_col):
    #     id = 0
    #     mother_ids = []
    #     current_child_cell = []
    #     current_child_cell.append(self.overall_cells_split(conditions, data_arr, target_col))
    #     mother_tree = self.tree_wrapper(id, current_child_cell[0].l_data, current_child_cell[0].r_data, current_child_cell[0].l_cell,
    #                                              current_child_cell[0].r_cell, current_child_cell[0].condition)
    #     # looping each layer
    #     for layer in np.arange(1, self.max_layer):
    #         num_mother = 2 ** (layer-1)
    #
    #         # update mother tree
    #         for c in np.arange(num_mother):
    #             left_done = self.cell_satisfied(mother_tree[id]["left"])
    #             if left_done == False:
    #                 self.overall_cells_split(conditions, data_arr, target_col)
    #             reft_done = self.cell_satisfied(mother_tree[id]["right"])
    #     pass














        #     mother_ids = []
        #     next_child_cells = []
        #     next_trees = []
        #     for c in np.arange(num_mother):
        #         next_child_cells.append(self.overall_cells_split(conditions, current_child_cell[c].l_data, current_child_cell[c].l_cell))
        #         next_child_cells.append(self.overall_cells_split(conditions, current_child_cell[c].r_data, current_child_cell[c].r_cell))
        #         next_trees.append(self.tree_wrapper(id, next_child_cells[c].l_data, next_child_cells[c].r_data, next_child_cells[c].l_cell,
        #                                          next_child_cells[c].r_cell, next_child_cells[c].condition))
        #         mother_ids.append(id)
        #         id = id + 1
        #     for i, mother_id in enumerate(mother_ids):
        #         # mother_tree, mother_id, tree_label, direction = 'l'
        #         mother_tree = self.merge_tree(mother_tree, mother_id, trees[i], )
        # return None



