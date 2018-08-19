import matplotlib.pyplot as plt
from keras.datasets import cifar10
from scipy.misc import toimage
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def show_images(imgs, name):
    plt.figure(figsize=[16, 16])
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            plt.subplot2grid((4, 4), (i, j))
            plt.imshow(toimage(imgs[k]))
            k = k + 1
    plt.savefig(name)
    plt.show()


(x_train, y_train), _ = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype(np.float32) / 255.0
y_train = to_categorical(y_train, num_classes=10)

# show raw image
# show_images(x_train[:16], name="org.png")

# feature standardization
# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

# zca whitening
# datagen = ImageDataGenerator(zca_whitening=True, zca_epsilon=1e-6)

# random rotation
# datagen = ImageDataGenerator(rotation_range=90)

# random shift
# datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)

# random flip
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

# channel shift
datagen = ImageDataGenerator(channel_shift_range=0.3)

datagen.fit(x_train)

feat_std_imgs = datagen.flow(x_train, y_train, batch_size=16, shuffle=False).next()[0]
show_images(feat_std_imgs, name="channel_shift.png")
