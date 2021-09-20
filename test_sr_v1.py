from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import gzip 
import time 
from scipy import signal

"""
[This code is a testing version of the tensorflow superesolution code ( mostly run on Lambda machines) ]

Returns:
    [type] -- [a pickle file containing the trained CNN predictions]
"""


def cnn_extract_data(filename, num_images, IMAGE_WIDTH):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data


def cnn_extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def cnn_convolution(image, filt, bias, stride=1):

    (num_filter, num_channel_filter, filter_size,_) = filt.shape  # filter dimensions
    num_channels, image_size_x, image_size_y = image.shape  # image dimensions

    out_dim_x = int((image_size_x - filter_size)/stride) + 1  # calculate output dimensions
    out_dim_y = int((image_size_y - filter_size)/stride) + 1  # calculate output dimensions  as per cs231n 


    assert num_channels == num_channel_filter, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((num_filter, out_dim_x, out_dim_y))
    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(num_filter):
        curr_y = out_y = 0
        while curr_y + filter_size <= image_size_x:
            curr_x = out_x = 0
            while curr_x + filter_size <= image_size_y:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y+filter_size, curr_x:curr_x+filter_size]) + bias[curr_f]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return out


def cnn_convolution_numpy(image, filt, bias, stride=1):
    
    (num_filter, num_channel_filter, filter_size,_) = filt.shape  # filter dimensions
    num_channels, image_size_x, image_size_y = image.shape  # image dimensions

    out_dim_x = int((image_size_x - filter_size)/stride) + 1  # calculate output dimensions
    out_dim_y = int((image_size_y - filter_size)/stride) + 1  # calculate output dimensions  as per cs231n 

    assert num_channels == num_channel_filter, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((num_filter, out_dim_x, out_dim_y))

    # out  = signal.fftconvolve(image, filt, mode='same')
    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(num_filter):
        curr_y = out_y = 0
        while curr_y + filter_size <= image_size_x:
            curr_x = out_x = 0
            while curr_x + filter_size <= image_size_y:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y+filter_size, curr_x:curr_x+filter_size]) + bias[curr_f]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

    return out


def cnn_maxpool(image, pool_size=2, stride=2):

    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - pool_size)/stride)+1
    w = int((w_prev - pool_size)/stride)+1

    pooled_image = np.zeros((n_c, h, w))
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + pool_size <= h_prev:
            curr_x = out_x = 0
            while curr_x + pool_size <= w_prev:
                pooled_image[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+pool_size, curr_x:curr_x+pool_size])
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return pooled_image

# 
def cnn_softmax(X):
    z = np.exp(X)
    return z/np.sum(z)


def cnn_predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=2, pool_f=2, pool_s=2):
    '''
    Make predictions with trained filters/weights. 
    '''
    t1 = time.time()
    # conv1 = cnn_convolution(image, f1, b1, conv_s)  # first convolution 
    conv1 = cnn_convolution_numpy(image, f1, b1, conv_s)  # using numpy 


    conv1[conv1 <= 0] = 0  # relu activation
    
    t2 = time.time()
    print('time taken for each convolution', (t2-t1))
    
    conv2 = cnn_convolution(conv1, f2, b2, conv_s)  # second convolution 
    conv2[conv2 <= 0] = 0  # ReLU 

    pooled = cnn_maxpool(conv2, pool_f, pool_s)  # maxpooling 
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten 

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # ReLU 

    out = w4.dot(z) + b4  # second dense layer
    probs = cnn_softmax(out)  # output softmax for class probabilities 
    
    return np.argmax(probs), np.max(probs)


if __name__ == '__main__':

    # create random filters, weights and bias to time it 

    f1 = np.random.randn(16,1,3,3)
    f2 = np.random.randn(32, 16, 3, 3)
    w3 = np.random.randn(128, 927360)
    w4 = np.random.randn(10, 128)
    b1 = np.random.randn(16,1)
    b2 = np.random.randn(32,1)
    b3 = np.random.randn(128,1)
    b4 = np.random.randn(10,1)


    test_examples = 100 
    X = np.random.randn(100, 1, 256,464)
    y  = np.arange(100)



    # save_path = 'weights_two_layer_cnn_params.pkl'
    # params, cost = pickle.load(open(save_path, 'rb'))
    # [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # # Get test data
    # test_examples =10000
    # img_row = 28
    # img_col = 28 
    num_classes = 10 

    # X = cnn_extract_data('t10k-images-idx3-ubyte.gz', test_examples, img_row)
    # y_dash = cnn_extract_labels('t10k-labels-idx1-ubyte.gz', test_examples).reshape(test_examples, 1)
    # # Normalize the data, linear normalziation is ok for MNIST, CIFAR. 
    # X-= int(np.mean(X)) 
    # X/= int(np.std(X)) 
    # test_data = np.hstack((X,y_dash))

    # X = test_data[:,0:-1]
    # X = X.reshape(len(test_data), 1, img_row, img_col)
    # y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(num_classes)]
    digit_correct = [0 for i in range(num_classes)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = cnn_predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    # print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(num_classes)
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()
