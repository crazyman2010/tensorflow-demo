import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageColor
import PIL.ImageFont


def download(url, max_dim=(256, 256)):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    img = img.convert("RGB")
    if max_dim:
        img.resize(max_dim)
    return img


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               class_name,
                               thickness=4):
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness, fill=color)
    draw.text((left, top),
              class_name,
              fill=color,
              font=font)


IMAGE_SHAPE = (720, 1280)
origin_image = download(
    "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1580305242309&di=46c9e5dfa8ea1568993ead5223156339&imgtype=0&src=http%3A%2F%2Fbpic.588ku.com%2Felement_origin_min_pic%2F16%2F11%2F22%2F3b8cc33493283cb885e826d5a212ef31.jpg",
    IMAGE_SHAPE)
img = np.array(origin_image) / 255.0
img = tf.cast(img, tf.float32)
images = tf.expand_dims(img, axis=0)

detector_url = "https://hub.tensorflow.google.cn/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(detector_url).signatures['default']
detection_output = detector(images)
print(detection_output)
detection_class_entities = detection_output['detection_class_entities']
detection_boxes = detection_output['detection_boxes']
detection_scores = detection_output['detection_scores']

i = 0
colors = list(PIL.ImageColor.colormap.values())
font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 36)
for score in detection_scores:
    if score > 0.50:
        color = colors[i + 20 % len(colors)]
        draw_bounding_box_on_image(origin_image,
                                   detection_boxes[i][0], detection_boxes[i][1],
                                   detection_boxes[i][2], detection_boxes[i][3],
                                   color, font,
                                   detection_class_entities[i].numpy().decode("ascii") + '('
                                   + str(round(score.numpy() * 100, 2)) + '%)')
    i = i + 1
plt.figure(figsize=(28, 10))
plt.grid(False)
plt.imshow(origin_image)
plt.show()
