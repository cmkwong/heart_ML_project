from lib import data, model, common, ML


PATH = "/home/chris/projects/Kaggle/heart_200719/data/heart.csv"
df = data.read_csv(PATH)
feature_list = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
type_list = ["num","mc","mc","num","num","mc","mc","num","mc","num","mc","mc","mc"]
decisionTree = ML.DecisionTree(0.05)

data_arr, target_col = (df.values)[:,:13], (df.values)[:,13]
target_col.reshape(target_col.shape[0], )

conditions = decisionTree.condition_build(data_arr, feature_list, type_list)

tree = decisionTree.build_tree(conditions, data_arr, target_col)
print(tree)
print("ok")