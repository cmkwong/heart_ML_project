from lib import data, model, common, ML, visualize
import numpy as np
import graphviz
from sklearn import tree

READ_SAME = False
MAX_DATA_COL = 13
PATH = "/home/chris/projects/Kaggle/heart_200719/data/heart.csv"
TYPE_LIST = ["num","mc","mc","num","num","mc","mc","num","mc","num","mc","mc","mc"]
decisionTree = ML.DecisionTree(tolerance=0.00, min_element=1, max_depth=100)

if READ_SAME:
    df = data.read_csv("/home/chris/projects/Kaggle/heart_200719/data/heart_debug.csv")
    seed_arr = data.df2array(df)

    # get the feature name list
    feature_list = []
    for i, key in enumerate(df.keys()):
        feature_list.append(key)
else:
    # read the df
    df = data.read_csv(PATH)

    # get the feature name list
    feature_list = []
    for i, key in enumerate(df.keys()):
        feature_list.append(key)

    # shuffle data
    seed_arr = data.shuffle(data.df2array(df))

    # output csv after shuffle
    data.out_csv(path="../data/heart_debug.csv", df=data.array2df(seed_arr, feature_list))

print("-------------------------------sklearn Decision Tree-------------------------------")
# split the data into training set and testing set
train_set, test_set = data.split_data(seed_arr, percentage=0.8)

data_arr, target_col = (train_set)[:, :MAX_DATA_COL], (train_set)[:, MAX_DATA_COL]

# training sklearn random forest
clf = tree.DecisionTreeClassifier()
clf.fit(data_arr, target_col)

# testing data
data_arr, target_col = (test_set)[:, :MAX_DATA_COL], (test_set)[:, MAX_DATA_COL]
predicted_labels = []
for index in np.arange(data_arr.shape[0]): # looping each row
    test_row = data_arr[index]
    label = clf.predict([(test_row)])[0]
    predicted_labels.append(label)
accuracy = decisionTree.calc_acc(np.array(predicted_labels), np.array(target_col))
print("The sklearn accuracy is %.2f" % (accuracy), "%")

# visualise the tree
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

print("-------------------------------Decision Tree-------------------------------")
# split the data into training set and testing set
train_set, test_set = data.split_data(seed_arr, percentage=0.8)
# training
# data array and target array
data_arr, target_col = (train_set)[:,:MAX_DATA_COL], (train_set)[:,MAX_DATA_COL]
# re-label the target_col (because the origin label is number)
target_col = decisionTree.num2label(target_col)

# building all conditions for building tree
conditions = decisionTree.condition_build(data_arr, feature_list[:-1], TYPE_LIST)
show_tree, prod_tree = decisionTree.build_tree(conditions, data_arr, target_col)
# print(show_tree)

# visualize the tree
drawer = visualize.Drawer(graphviz.Digraph(), target_type=type('A'))
drawer.interpret(show_tree)
drawer.dot.format = 'png'
drawer.dot.render('image_example', view=True)

# testing
data_arr, target_col = (test_set)[:,:MAX_DATA_COL], (test_set)[:,MAX_DATA_COL]
# re-label the target_col (because the origin label is number)
target_col = decisionTree.num2label(target_col)

predicted_labels = []
for index in np.arange(data_arr.shape[0]): # looping each row
    test_row = data_arr[index]
    label = decisionTree.predict(test_row, prod_tree)
    predicted_labels.append(label)

accuracy = decisionTree.calc_acc(np.array(predicted_labels), np.array(target_col))
print("The accuracy is %.2f" % (accuracy), "%")







