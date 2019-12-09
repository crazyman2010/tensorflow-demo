import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("boston.csv")
df = df.values
df = np.array(df)
# 归一化数据
for i in range(12):
    df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
# 留10行作为测试数据， 前0-11列是参数， 第12列是平均房价
x_train = df[10:, :12]
y_train = df[10:, 12]
x_test = df[0:10, :12]
y_test = df[0:10, 12]

model = tf.keras.Sequential(
    tf.keras.layers.Dense(1, input_shape=(12,))
)
model.compile(optimizer="sgd", loss="mse")

model.fit(x_train, y_train, batch_size=10, verbose=2, epochs=100)
# 评估损失， 不太好， 在20+
model.evaluate(x_train, y_train)
# 使用测试数据来看预测对比
for i in range(10):
    print("预测：", model.predict(x_test[i:i + 1])[0], "实际：", y_test[i:i + 1])
