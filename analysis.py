from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np

def cnn_model(input_shape):
    # We use the sequential model
    model = Sequential()
    # A 2-D convolutional layer
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    # Pooling layer with pool size and strides = (2,2)
    model.add(MaxPooling2D(pool_size=2))
    # Add a dropout layer to avoid overfitting
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    # Flattens the output for normal neural network layer
    model.add(Flatten())
    # Add a normal densely connected neural network layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    # Output layer with 10 outputs, uses softmax classifier
    model.add(Dense(10, activation='softmax'))

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(), loss=categorical_crossentropy,
                  metrics=['accuracy'])

    return model

def main():

    classes = 10
    input_shape = (28, 28, 1)
    epochs = 20
    batch_size = 100

    # Load data, training and testing sets
    # The data is already shuffled at this point
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data to include channel dimension, required for Keras
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Scale the values between 0 and 1
    x_train = x_train / 255
    x_test = x_test / 255

    # Perform one-hot encoding for output
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # Define and compile the cnn model
    model = cnn_model(input_shape)

    # Fit the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))

    # Print the final metrics
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Print confusion matrix
    y_predicted = model.predict(x_test)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_actual = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_actual, y_predicted)
    print(cm)

if __name__ == '__main__':
    main()
