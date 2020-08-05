from lib import data, model, common, ML, visualize
import numpy as np
import graphviz
from sklearn.ensemble import RandomForestClassifier

PATH = "/home/chris/projects/Kaggle/heart_200719/data/heart.csv"
TYPE_LIST = ["num","mc","mc","num","num","mc","mc","num","mc","num","mc","mc","mc"]
MAX_DATA_COL = 13

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

randomForest = ML.RandomForest(tolerance=0.05, min_element=10, max_depth=100, num_col_blocked_each_step=2, conditions_keep_prob_each_step=0.8, OOB_percentage=0.2)

print("-------------------------------sklearn Random Forest-------------------------------")
data_arr, target_col = (train_set)[:, :MAX_DATA_COL], (train_set)[:, MAX_DATA_COL]

# training sklearn random forest
clf = RandomForestClassifier(n_estimators=200, max_depth=100, criterion="entropy" , random_state=0)
clf.fit(data_arr, target_col)

# testing data
data_arr, target_col = (test_set)[:, :MAX_DATA_COL], (test_set)[:, MAX_DATA_COL]
predicted_labels = []
for index in np.arange(data_arr.shape[0]): # looping each row
    test_row = data_arr[index]
    label = clf.predict([(test_row)])[0]
    predicted_labels.append(label)
accuracy = randomForest.calc_acc(np.array(predicted_labels), np.array(target_col))
print("The sklearn accuracy is %.2f" % (accuracy), "%")

print("-------------------------------Random Forest-------------------------------")
tolerance = 0.05
min_element = 10
max_depth = 100
num_col_blocked_each_step = 2
conditions_keep_prob_each_step = 0.8
OOB_percentage = 0.2
NUM_BOOTSTRAP_DATA = 200

train_set, test_set = data.split_data(seed_arr, percentage=0.8)
bootstrap_train_sets, OOB_indexs, show_trees, prod_trees = [], [], [], []

for count in np.arange(NUM_BOOTSTRAP_DATA):
    bootstrap_train_set, OOB_index = randomForest.bootstrap_data(seed_arr)
    bootstrap_train_sets.append(bootstrap_train_set)
    OOB_indexs.append(OOB_index)

    # data array and target array
    data_arr, target_col = (bootstrap_train_set)[:, :MAX_DATA_COL], (bootstrap_train_set)[:, MAX_DATA_COL]
    # re-label the target_col (because the origin label is number)
    target_col = randomForest.num2label(target_col)

    # building the conditions and stored trees
    conditions = randomForest.condition_build(data_arr, feature_list, TYPE_LIST)
    show_tree, prod_tree = randomForest.build_tree(conditions, data_arr, target_col, drop_variable_each_step=True)
    show_trees.append(show_tree)
    prod_trees.append(prod_tree)
    # if (count % 50) == 0:
    #     print(count, " progress")

data_arr, target_col = (seed_arr)[:, :MAX_DATA_COL], (seed_arr)[:, MAX_DATA_COL]
target_col = randomForest.num2label(target_col)
error = randomForest.OOB_error(data_arr, target_col, prod_trees, OOB_indexs)
accuracy = 100 - error
print("----------------- Accuracy: %.2f" % accuracy, "% -----------------")



# tolerances = [0, 0.05, 0.1, 0.15, 0.20, 0.25]
# conditions_keep_prob_each_steps = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
# num_col_blocked_each_steps = [1,2,3,4,5,6,7,8,9,10]
# OOB_percentages = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# NUM_BOOTSTRAP_DATAS = [100,200,300,400,500,600]