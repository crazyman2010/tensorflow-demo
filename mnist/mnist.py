import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 获取训练与测试数据
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
print("训练集大小：", images_train.shape)
print("测试集大小：", images_test.shape)

# 显示一张图片看看
# plt.imshow(images_train[0])
# plt.show()

model = tf.keras.Sequential(
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
)
