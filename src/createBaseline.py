from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm

def createBaseline(learning_rate, l2_factor, dropout_factor):
    # Create CNN from the paper.
    model = Sequential()

    # Starts with input 2D convolutional layer with input shape matching the images shaper and
    # l2 regularization enabled.
    model.add(Conv2D(96, (11, 11), input_shape=(512, 512, 1), kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

    # Use batch normalization for the first layer, then do activation and pooling.
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3)))

    # Add next 2D convolutional layer with smaller filter size and the same regularization as
    # in the first layer.
    model.add(Conv2D(384, (5, 5), kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3)))

    # The next three layers are almost the same, having similar approach to VGG19 with multiple
    # convolutional layers connected in sequence before pooling and all having 3x3 filters
    # (a.k.a. kernels). The filter count decreases slightly to account for compression of
    # information the farther it goes through the network. All layers use l2 regularization.
    model.add(Conv2D(384, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

    # Apply activation and pooling.
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3)))

    # Flatten the outputs form 2 dimensional layers to be ready for fully-connected layers.
    model.add(Flatten())

    # Create fully connected layers with dropout layers in-between.
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_factor))
    model.add(Dense(1, activation='sigmoid', bias_constraint=MinMaxNorm(min_value=0.1, max_value=0.9)))


    # Compile the model with Adam optimizer and Binary Crossentropy loss function for binary
    # classification. The metrics used for this model are accuracy, mean-square-error,
    # precision and recall.
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', 'mse', Precision(), Recall()])

    return model
