import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 获取训练与测试数据
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train / 255.0, images_test / 255.0
print("训练集大小：", images_train.shape)
print("测试集大小：", images_test.shape)

# 显示一张图片看看
# plt.imshow(images_train[0])
# plt.show()

# 将二维图片展开成一维数组
# 定义一个128个节点的神经网络层
# 定义一个10节点的softmax层，用于输出10个可能的分类
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 开始训练
model.fit(images_train, labels_train, epochs=10)

# 评估误差
model.evaluate(images_test, labels_test, verbose=2)
