import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Local imports
from loading import loadLargePolypImages, loadMediumPolypImages, loadRandomNoPolypImages

# Load dataset
large_polyp_images = loadLargePolypImages()
large_polyp_pixel_data = [im.pixel_array for im in large_polyp_images]
large_count = len(large_polyp_images)

medium_polyp_images = loadMediumPolypImages()
medium_polyp_pixel_data = [im.pixel_array for im in medium_polyp_images]
medium_count = len(medium_polyp_images)

no_polyp_images = loadRandomNoPolypImages()
no_polyp_pixel_data = [im.pixel_array for im in no_polyp_images]

total_count = large_count + medium_count
# total_count = large_count
print("Total images:", total_count)

# Trip to the size of the confirmed polyp images.
no_polyp_pixel_data = no_polyp_pixel_data[:total_count]

# plt.imsave("./sample.jpg", large_polyp_pixel_data[1], cmap=plt.cm.bone)

X = large_polyp_pixel_data + medium_polyp_pixel_data
# X = large_polyp_pixel_data
y = [(1, 0)]*total_count

X = X + no_polyp_pixel_data
y = y + [(0, 1)]*len(no_polyp_pixel_data)

print("Total: X =", len(X), "; y =", len(y))

X = np.array([
    np.array([
        x.reshape(512, 512, 1),
        x.reshape(512, 512, 1),
        x.reshape(512, 512, 1)
    ]).reshape(512, 512, 3) for x in X
])
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# TODO: Reshape each sample into into (512, 512, 3)

# Create VGG16 model, and train it on the data.

X = preprocess_input(X)

base_model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(512, 512, 3),
    pooling=None, classes=2, classifier_activation='softmax'
)

# vgg16.summary()

feature_batch = base_model(X)
print(feature_batch.shape)

flat1 = layers.GlobalAveragePooling2D()(base_model(X))
class1 = layers.Dense(2048, activation='relu')(flat1)
class2 = layers.Dense(2048, activation='relu')(class1)
output = layers.Dense(10, activation='softmax')(class2)
# define new model
model = models.Sequential([
    base_model,
    flat1,
    class1,
    class2,
    output
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(
              #     from_logits=True),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X, y, epochs=150)
