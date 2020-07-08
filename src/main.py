import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np

# Local imports
from loading import loadBinaryDataset

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadBinaryDataset()

# Trip to the size of the confirmed polyp images.
# no_polyp_pixel_data = no_polyp_pixel_data[:total_count] 

# exit()

# Create CNN
model = models.Sequential()
model.add(layers.Conv2D(96, (11, 11), activation='relu', input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.LayerNormalization())
model.add(layers.Conv2D(384, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.LayerNormalization())
model.add(layers.Conv2D(384, (3, 3), activation='relu'))
model.add(layers.Conv2D(384, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))

model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# exit()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(
              #     from_logits=True),
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Sizes:", X_train.shape, y_train.shape)
# exit()

# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

# exit()

history = model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('./accuracy.png')

predictions = model.predict(X_test, batch_size=4)

predictions = [p[0] for p in predictions]

print(len(y_test))
print(len(predictions))
cf = confusion_matrix(y_test, predictions)

print(cf)

# TODO:
# Confusion matrix
# Implement several general purpose classifier using transfer learning.
# Write up
