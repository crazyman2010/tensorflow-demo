import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 150
IMG_WIDTH = 150
image_generator = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=180)
image_generator = image_generator.flow_from_directory(batch_size=32,
                                                      directory='data',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode='sparse')

# augmented_images = [image_generator[0][0][0] for i in range(5)]
# plt.figure(figsize=(10, 1))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(augmented_images[i])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(27, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit_generator(image_generator, epochs=100)


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


augmented_images = [image_generator[0][0][0] for i in range(5)]
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmented_images[i])
    plt.xticks([])
    plt.yticks([])
    class_index = np.argmax(model.predict(tf.expand_dims(augmented_images[i], axis=0)))
    plt.xlabel(get_key(image_generator.class_indices, class_index)[0])
plt.show()
