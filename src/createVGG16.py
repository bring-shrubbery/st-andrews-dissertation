from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2

# Create VGG16 structure
def createVGG16(learning_rate, l2_factor, dropout_factor):
    model = Sequential()

    # Add layers
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(512, 512, 1), kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with Adam optimizer and Binary Crossentropy loss function for binary
    # classification. The metrics used for this model are accuracy, mean-square-error,
    # precision and recall.
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', 'mse', Precision(), Recall()])

    return model
