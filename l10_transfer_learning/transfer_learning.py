import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image

classifier_url = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

grace_hopper = tf.keras.utils.get_file('image.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper) / 255.0

result = classifier.predict(grace_hopper[np.newaxis, ...])
predict_class = np.argmax(result[0], axis=-1)
print(predict_class)

classifier_labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                                 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
classifier_labels = np.array(open(classifier_labels_path).read().splitlines())
print('Prediction: ' + classifier_labels[predict_class])
# plt.imshow(grace_hopper)
# plt.xlabel('Prediction: ' + classifier_labels[predict_class])
# plt.show()

data_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flower_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
flower_image_data = flower_image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for flower_image_batch, flower_label_batch in flower_image_data:
    print("Image batch shape: ", flower_image_batch.shape)
    print("Label batch shape: ", flower_label_batch.shape)
    break

# flower_result = classifier.predict(flower_image_batch)
# plt.figure(figsize=(10, 9))
# plt.subplots_adjust(hspace=0.5)
# for i in range(30):
#     plt.subplot(6, 5, i + 1)
#     plt.yticks([])
#     plt.xticks([])
#     plt.xlabel(classifier_labels[np.argmax(flower_result[i], axis=-1)])
#     plt.imshow(flower_image_batch[i])
# plt.show()

feature_extractor_url = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=IMAGE_SHAPE + (3,))
feature_batch = feature_extractor_layer(flower_image_batch)
print(feature_batch.shape)
