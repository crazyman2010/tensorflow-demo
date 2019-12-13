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

# 未加载时评估
loss, acc = model.evaluate(images_test, labels_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 设置保存点信息
checkpoint_path = "checkpoints/cp.ckpt"
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(images_test, labels_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
