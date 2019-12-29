import json
import os

import tensorflow as tf

# 获取训练与测试数据
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train / 255.0, images_test / 255.0


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(images_train, labels_train, epochs=3)

# task type: "chief", "worker", "evaluator", "ps"
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_WORKERS = 2
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(images_train, labels_train, epochs=3)
