import PIL.Image
import os
import matplotlib.pyplot as plt

srcImage = PIL.Image.open('src1.png')
rows = 3
cols = 9
width, height = srcImage.size
uw = width / cols
uh = height / rows
imageList = []
for i in range(rows):
    for j in range(cols):
        img = srcImage.crop((j * uw, i * uh, (j + 1) * uw, (i + 1) * uh))
        imageList.append(img)

# plt.figure(figsize=(10, 10))
# for i in range(rows * cols):
#     plt.subplot(rows, cols, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(imageList[i])
# plt.show()

if not os.path.isdir('data'):
    os.mkdir('data')
os.chdir('data')
index = 1
for img in imageList:
    class_path = 'class_' + str(index)
    if not os.path.isdir(class_path):
        os.mkdir(class_path)
    img.save(os.path.join(class_path, str(index) + '.png'), 'PNG')
    index += 1
