from lib import data, model, common, ML
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
MAX_DATA_COL = 13
PATH = "/home/chris/projects/Kaggle/heart_200719/data/heart.csv"
TYPE_LIST = ["num","mc","mc","num","num","mc","mc","num","mc","num","mc","mc","mc"]
decisionTree = ML.DecisionTree(0.05)

# read the df
df = data.read_csv(PATH)

# get the feature name list
feature_list = []
for i, key in enumerate(df.keys()):
    feature_list.append(key)
    if i == (MAX_DATA_COL-1):
        break

# shuffle data
arr = data.shuffle(data.df2array(df))
# split the data into training set and testing set
train_set, test_set = data.split_data(arr, percentage=0.8)

# training
# data array and target array
data_arr, target_col = (train_set)[:,:MAX_DATA_COL], (train_set)[:,MAX_DATA_COL]
# re-label the target_col (because the origin label is number)
target_col = decisionTree.num2label(target_col)

# building all conditions for building tree
conditions = decisionTree.condition_build(data_arr, feature_list, TYPE_LIST)
show_tree, prod_tree = decisionTree.build_tree(conditions, data_arr, target_col)
print(show_tree)

# testing
predicted_labels = []
data_arr, target_col = (test_set)[:,:MAX_DATA_COL], (test_set)[:,MAX_DATA_COL]
# re-label the target_col (because the origin label is number)
target_labels = decisionTree.num2label(target_col)
# get the label set
label_set = decisionTree.get_set(target_labels)

for index in np.arange(data_arr.shape[0]): # looping each row
    test_tree = prod_tree.copy()
    c = 0
    loop = True
    test_row = data_arr[index]
    while(loop == True):
        condition = list(test_tree.keys())[0]
        selection = condition.judge(test_row)
        test_tree = test_tree[condition][selection]
        if prod_tree == 'A':
            print("stop")
        if type(test_tree) is not dict:
            predicted_label = test_tree
            predicted_labels.append(predicted_label)
            loop = False

accuracy = decisionTree.calc_acc(np.array(predicted_labels), np.array(target_labels))
print("The accuracy is %.2f" % (accuracy), "%")




