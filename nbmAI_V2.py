import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('resources/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Reshape the input data to (batch_size, height, width, channels)
X_train = X_train.T.reshape(-1, 28, 28, 1)  # Assuming grayscale images (1 channel)
X_dev = X_dev.T.reshape(-1, 28, 28, 1) # Assuming grayscale images (1 channel)


#print(Y_train)
#print(X_train)

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # Initialize weights (filters) and biases
        self.weights = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.01
        self.bias = np.zeros(out_channels)

        # Store parameters
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Cache variables for backprop
        self.cache = None  # Store input for backprop

    def forward(self, input_data):
        # Implement vectorized convolution using im2col for efficiency

        # Implement padding if needed
        if self.padding > 0:
            input_data = np.pad(input_data,
                                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                                mode='constant')

        batch_size, h_in, w_in, c_in = input_data.shape
        f_h, f_w, c_in, c_out = self.weights.shape

        h_out = (h_in - f_h) // self.stride + 1
        w_out = (w_in - f_w) // self.stride + 1

        # Use np.lib.stride_tricks.as_strided for vectorized convolution
        windows = np.lib.stride_tricks.as_strided(input_data,
                                                  shape=(batch_size, h_out, w_out, f_h, f_w, c_in),
                                                  strides=(input_data.strides[0], self.stride * input_data.strides[1],
                                                           self.stride * input_data.strides[2],
                                                           input_data.strides[1], input_data.strides[2],
                                                           input_data.strides[3]))

        # Reshape for matrix multiplication
        windows = windows.reshape(batch_size, h_out, w_out, -1)  # Flatten the kernel dimensions
        weights_flat = self.weights.reshape(-1, c_out)

        # Perform matrix multiplication (convolution)
        output = np.tensordot(windows, weights_flat, axes=((3,), (0,)))  #

        # Add bias
        output += self.bias

        # Store input for backprop
        self.cache = input_data

        return output

    def backward(self, grad_output):
        # Implement backpropagation for the convolutional layer
        # This is more involved and requires calculating gradients for weights, biases, and input.
        pass


def init_params():
    # Initialize the convolutional layer
    conv_layer = ConvLayer(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Example: 16 filters, 3x3 kernel, 1 pixel padding

    # Calculate the output size of the convolutional layer
    # This will determine the input size of the subsequent fully connected layer
    # Assuming input size of 28x28
    h_out = (28 + 2 * conv_layer.padding - conv_layer.weights.shape[0]) // conv_layer.stride + 1
    w_out = (28 + 2 * conv_layer.padding - conv_layer.weights.shape[1]) // conv_layer.stride + 1

    # Initialize weights and biases for the second layer (now a fully connected layer)
    # The input size is the flattened output of the convolutional layer
    fc_input_size = h_out * w_out * conv_layer.out_channels
    W2 = np.random.rand(10, fc_input_size) - 0.5  # Output size of 10 for digits
    b2 = np.random.rand(10, 1) - 0.5

    return conv_layer, W2, b2



def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(conv_layer, W2, b2, X):
    # Pass input through the convolutional layer
    conv_output = conv_layer.forward(X)

    # Flatten the convolutional output
    conv_output_flat = conv_output.reshape(conv_output.shape[0], -1).T # Flatten to (input_features, batch_size)

    # Pass flattened output through the fully connected layer
    Z2 = W2.dot(conv_output_flat) + b2
    A2 = softmax(Z2)

    return conv_output, Z2, A2



def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(conv_output, Z2, A2, W2, X, Y, conv_layer):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(conv_output.reshape(conv_output.shape[0], -1)) # Reshape conv_output for dot product with dZ2
    db2 = 1 / m * np.sum(dZ2)

    # Unflatten the gradient from the fully connected layer
    grad_conv_flat = W2.T.dot(dZ2)
    grad_conv = grad_conv_flat.T.reshape(conv_output.shape) # Reshape back to convolutional output shape

    # Backpropagate through the convolutional layer
    # This requires implementing the conv_layer.backward(grad_conv) method
    # You'll get dX, dWconv, dbconv from this method.
    pass



def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    conv_layer, W2, b2 = init_params() # Get convolutional layer instance

    for i in range(iterations):
        conv_output, Z2, A2 = forward_prop(conv_layer, W2, b2, X) # Pass conv_layer to forward_prop

        # Implement backward_prop for the convolutional layer
        # dX, dWconv, dbconv = backward_prop(conv_output, Z2, A2, W2, X, Y, conv_layer) # Implement conv_layer.backward

        # Update weights and biases for the fully connected layer
        # W2, b2 = update_params(W2, b2, dW2, db2, alpha) # Need to calculate dW2, db2 from backward_prop

        # Update weights and biases for the convolutional layer
        # conv_layer.weights -= alpha * dWconv
        # conv_layer.bias -= alpha * dbconv

        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return

conv_layer, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


def make_predictions(conv_layer, W2, b2, X):
    _, _, _, A2 = forward_prop(conv_layer, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, conv_layer, W2, b2, X_test, Y_test):
    """
    Tests the prediction of a single image using the trained model with a convolutional layer.

    Args:
        index (int): The index of the image in the test set.
        conv_layer (ConvLayer): The trained convolutional layer.
        W2 (np.array): Weights of the fully connected layer.
        b2 (np.array): Biases of the fully connected layer.
        X_test (np.array): Test image data (batch_size, height, width, channels).
        Y_test (np.array): True labels of the test images.
    """
    # Select the image and reshape for the convolutional layer
    current_image = X_test[index:index+1] # Take a slice to keep the batch dimension

    # Pass the image through the forward pass of the network
    # You'll need to define a make_predictions function that incorporates the forward pass
    conv_output, Z2, A2 = forward_prop(conv_layer, W2, b2, current_image) # Use your forward_prop function

    # Get the prediction from the output of the fully connected layer
    prediction = np.argmax(A2)

    # Get the true label
    label = Y_test[index]

    print("Prediction: ", prediction)
    print("Label: ", label)

    # Reshape and display the image
    current_image_display = current_image.reshape((28, 28)) * 255 # Assuming grayscale images
    plt.gray()
    plt.imshow(current_image_display, interpolation='nearest')
    plt.title(f"Prediction: {prediction}, Label: {label}")
    plt.show()

# Test a prediction on a number,
test_prediction(0, conv_layer, W2, b2, X_dev, Y_dev)


# Get predictions and accuracy of them
#dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
#print(get_accuracy(dev_predictions, Y_dev))