import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 图片下载地址
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# 图片下载函数
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail([max_dim, max_dim])
    return np.array(img)


def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# 损失计算函数
def calc_loss(img, model):
    # 插入一个维度， 变成[1,img]
    img_batch = tf.expand_dims(img, axis=0)
    # 模型有多个输出: [<tf.Tensor 'model/mixed1/concat:0' shape=(1, None, None, 288) dtype=float32>,
    # <tf.Tensor 'model/mixed2/concat:0' shape=(1, None, None, 288) dtype=float32>]
    layer_activations = model(img_batch)

    losses = []
    for act in layer_activations:
        # 单个输出的损失是所有维度的平均值
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    # 总损失是所有输出的损失的和
    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32)
        )
    )
    def __call__(self, img, steps, step_size):
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)
            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            # 将张量裁剪到-1,1的范围内
            img = tf.clip_by_value(img, -1, 1)
        return loss, img


# 下载图片，并设置大小为500x500
original_img = download(url, max_dim=500)
# plt.imshow(original_img)
# plt.show()

# 定义模型
# 使用 inceptionV3模型（权重集为imagenet）， 使用其中mixed1和mixed2作为输出
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
names = ['mixed1', 'mixed2']
layers = [base_model.get_layer(name).output for name in names]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
# print(dream_model.summary())

# 对图像进行预处理,处理后图像在[-1,1]区间内
img = tf.keras.applications.inception_v3.preprocess_input(original_img)
# 将图像数组转换成张量
img = tf.convert_to_tensor(img)
# 计算梯度，并把梯度加到图像上面
deep_dream = DeepDream(dream_model)
loss, dream_img = deep_dream(img, 100, tf.constant(0.01))
plt.imshow(dream_img)
plt.show()
