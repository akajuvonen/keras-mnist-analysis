from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

N_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
EPOCHS = 10

# Load data, training and testing sets
# The data is already shuffled at this point
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Scale the values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Perform one-hot encoding for output
y_train = to_categorical(y_train, N_CLASSES)
y_test = to_categorical(y_test, N_CLASSES)

# We use the sequential model
model = Sequential()
# A 2-D convolutional layer
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                 input_shape=INPUT_SHAPE))
# Pooling layer with pool size and strides = (2,2)
model.add(MaxPooling2D(pool_size=2))
# Flattens the output for normal neural network layer
model.add(Flatten())
# Output layer with 10 outputs, uses softmax classifier
model.add(Dense(10, activation='softmax'))

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(), loss=categorical_crossentropy,
              metrics=['accuracy'])
