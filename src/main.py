import tensorflow as tf

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

X = np.array([x.reshape(512, 512, 1) for x in X])
y = np.array(y)

# exit()


# Create CNN
model = models.Sequential()
model.add(layers.Conv2D(96, (11, 11),
                        activation='relu', input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
# model.add(layers.LayerNormalization())
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
# model.add(layers.LayerNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))  # Added
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3)))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(2))

print(model.summary())

# exit()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(
              #     from_logits=True),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
history = model.fit(X, y, epochs=50)

plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('./accuracy.jpg')

# TODO:
# Confusion matrix
# Fix 50% accuracy
# Batching
# Implement general purpose classifier using transfer learning.
