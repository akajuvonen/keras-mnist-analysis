from keras.utils import to_categorical
from keras.datasets import mnist

N_CLASSES = 10

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
