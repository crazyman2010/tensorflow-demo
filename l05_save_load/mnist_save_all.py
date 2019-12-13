import os

import tensorflow as tf

# 获取训练与测试数据
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train / 255.0, images_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 开始训练, 训练会自动保存权重
model.fit(images_train, labels_train, epochs=5,
          validation_data=(images_test, labels_test))

os.mkdir("models")
model.save("models/mnist_model.h5")
