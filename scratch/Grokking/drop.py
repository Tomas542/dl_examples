import numpy as np
import sys
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])

np.random.seed(1)

def relu(x: np.ndarray) -> np.ndarray:
    return (x >= 0) * x # returns x if x > 0

def relu2deriv(output: np.ndarray) -> np.ndarray:
    return output >= 0 # returns 1 for input > 0

batch_size = 100
alpha, iterations = (0.001, 300)
pixels_per_image, num_labels, hidden_size = (784, 10, 100)

weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size),((i+1)*batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
                correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))

                layer_2_delta = (labels[batch_start:batch_end]-layer_2)/batch_size
                layer_1_delta = layer_2_delta.dot(weights_1_2.T)* relu2deriv(layer_1)
                layer_1_delta *= dropout_mask

                weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
                weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
                
    if(j % 10 == 0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
                layer_0 = test_images[i:i+1]
                layer_1 = relu(np.dot(layer_0,weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        sys.stdout.write("\n" + \
                            "I:" + str(j) + \
                            " Test-Err:" + str(test_error/ float(len(test_images)))[0:5] +\
                            " Test-Acc:" + str(test_correct_cnt/ float(len(test_images)))+\
                            " Train-Err:" + str(error/ float(len(images)))[0:5] +\
                            " Train-Acc:" + str(correct_cnt/ float(len(images))))