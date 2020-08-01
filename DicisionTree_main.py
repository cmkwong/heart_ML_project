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
seed_arr = data.shuffle(data.df2array(df))
# split the data into training set and testing set
train_set, test_set = data.split_data(seed_arr, percentage=0.8)

print("-------------------------------Decision Tree-------------------------------")
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
data_arr, target_col = (test_set)[:,:MAX_DATA_COL], (test_set)[:,MAX_DATA_COL]
# re-label the target_col (because the origin label is number)
target_labels = decisionTree.num2label(target_col)
# get the label set
label_set = decisionTree.get_set(target_labels)

predicted_labels = []
for index in np.arange(data_arr.shape[0]): # looping each row
    test_row = data_arr[index]
    label = decisionTree.predict(test_row, prod_tree)
    predicted_labels.append(label)

accuracy = decisionTree.calc_acc(np.array(predicted_labels), np.array(target_labels))
print("The accuracy is %.2f" % (accuracy), "%")

print("-------------------------------Random Forest-------------------------------")
tolerances = [0.05, 0.1, 0.15, 0.20, 0.25]
keep_probabilities = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
OOB_percentages = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NUM_BOOTSTRAP_DATAS = [100,200,300,400,500,600,700,800]

for p1 in tolerances:
    for p2 in keep_probabilities:
        for p3 in OOB_percentages:
            for p4 in NUM_BOOTSTRAP_DATAS:
                print("tolerances: ", p1, " keep_probabilities: ", p2, " OOB_percentages: ", p3,
                      " NUM_BOOTSTRAP_DATAS: ", p4)
                randomForest = ML.RandomForest(tolerance=p1, num_col_blocked_each_step=1, keep_probability=p2)
                NUM_BOOTSTRAP_DATA = p4

                # getting 200 bootstrap dataset and building trees
                bootstrap_train_sets = []
                OOB_indexs = []
                show_trees = []
                prod_trees = []
                for count in np.arange(NUM_BOOTSTRAP_DATA):
                    bootstrap_train_set, OOB_index = randomForest.bootstrap_data(seed_arr, OOB_percentage=p3)
                    bootstrap_train_sets.append(bootstrap_train_set)
                    OOB_indexs.append(OOB_index)

                    # data array and target array
                    data_arr, target_col = (bootstrap_train_set)[:, :MAX_DATA_COL], (bootstrap_train_set)[:, MAX_DATA_COL]
                    # re-label the target_col (because the origin label is number)
                    target_col = decisionTree.num2label(target_col)

                    # building the conditions and stored trees
                    conditions = decisionTree.condition_build(data_arr, feature_list, TYPE_LIST)
                    show_tree, prod_tree = randomForest.build_tree(conditions, data_arr, target_col, drop_variable_each_step=True)
                    show_trees.append(show_tree)
                    prod_trees.append(prod_tree)
                    # if (count % 50) == 0:
                    #     print(count, " progress")

                data_arr, target_col = (seed_arr)[:, :MAX_DATA_COL], (seed_arr)[:, MAX_DATA_COL]
                target_col = decisionTree.num2label(target_col)
                error = randomForest.OOB_error(data_arr, target_col, prod_trees, OOB_indexs)
                accuracy = 100 - error
                print("----------------- Accuracy: %.2f" % accuracy, "% -----------------")






