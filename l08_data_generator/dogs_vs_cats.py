import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import math

train_dir = 'cats_and_dogs_filtered/train'
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_dir = 'cats_and_dogs_filtered/validation'
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

total_train_image_size = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))
total_validation_image_size = len(os.listdir(validation_cats_dir)) + len(os.listdir(validation_dogs_dir))
batch_size = 128
epochs = 2
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_images_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=[IMG_WIDTH, IMG_HEIGHT],
                                                           class_mode='binary')
validation_data_gen = validation_images_generator.flow_from_directory(batch_size=batch_size,
                                                                      directory=validation_dir,
                                                                      target_size=[IMG_WIDTH, IMG_HEIGHT],
                                                                      class_mode='binary')

class_names = ['cat', 'dog']
# next 返回  (x_train, y_train)
# sample_training_images, sample_training_labels = next(train_data_gen)
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(sample_training_images[i])
#     plt.xlabel(class_names[int(sample_training_labels[i])])
# plt.show()

model = tf.keras.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
#
history = model.fit_generator(train_data_gen,
                              epochs=epochs,
                              steps_per_epoch=total_train_image_size,
                              validation_data=validation_data_gen,
                              validation_steps=total_validation_image_size)
