import pandas as pd
import tensorflow as tf

dataset_path = tf.keras.utils.get_file("auto-mpg.data",
                                       "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?",
                          comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# print(dataset.head())
# print(dataset.tail())

# 删除无效数据
# print(dataset.isna().sum())
dataset = dataset.dropna()
# print(dataset.isna().sum())

# 将Origin转换成one-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

# 拆分数据为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 数据检查
# pair_grid=sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
