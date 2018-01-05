from keras.datasets import mnist

# Load data, training and testing sets
# The data is already shuffled at this point
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(max(x_train.any()))
