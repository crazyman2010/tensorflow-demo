import tensorflow as tf

# 获取训练与测试数据
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()

images_test = images_test / 255.0

model = tf.keras.models.load_model("models/mnist_model.h5")
loss, acc = model.evaluate(images_test, labels_test)
print("acc is ", acc)
