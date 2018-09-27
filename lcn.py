import cv2
import numpy as np
from numpy.linalg import norm
from scipy.signal.signaltools import convolve2d


def original_lcn(x):
    h, w = x.shape[:2]
    normed = np.zeros((h, w), np.float32)

    for i in range(h):
        for j in range(w):

            lowj  = max(0, j-1)
            highj = min(h, j+2)
            lowi  = max(0, i-1)
            highi = min(w, i+2)

            sliding_window = x[lowi:highi, lowj:highj]
            sliding_window_mean = np.mean(sliding_window)
            sliding_window_norm = norm(sliding_window)

            normed[i, j] = (x[i, j] - sliding_window_mean)
            if sliding_window_norm > 1:
                normed[i, j] /= sliding_window_norm

    return normed

def lcn(x):
    h, w = x.shape[:2]
    k = np.ones((3, 3))
    k /= 9
    meaned = convolve2d(x, k, mode = 'same')
    p = np.power(x, 2.0)

    s = convolve2d(p, np.ones((3, 3)), mode = 'same')
    s = np.sqrt(s)

    m =  x - meaned
    lcned = (m/s).reshape((h, w, 1))
    lcn_min = np.min(lcned)
    lcn_max = np.max(lcned)
    normed = (lcned - lcn_min) * (1/(lcn_max - lcn_min))

    return normed


# Load pickled data
import pickle
import numpy as np

training_file   = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file    = "./traffic-signs-data/test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

newshape = (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

newX = np.zeros(newshape, np.float32)

for i in range(len(X_train)):
    img = X_train[i]
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]

    newX[i] = lcn(y)

    if i == 3508:
        l = newX[i].copy()
        l *= 255
        cv2.imwrite('foobar.jpg', l)
        cv2.imwrite('foobarreal.jpg', img)


print(newX.shape)


    # y = np.matrix([[221, 235,  83,  40, 203,  25, 148, 250, 170],
    #            [183, 247, 252,  62, 185, 118,  98, 137,  18],
    #            [118, 199,  55,  79, 199,  87,  44, 132,  61],
    #            [134, 237, 136,  10,  43, 158, 247, 190,  95],
    #            [221, 145,  67,  37, 117, 140,   9, 118,  61],
    #            [ 95, 160, 102, 141, 240,  79, 240, 104, 221],
    #            [ 58, 162, 127, 192,  38,  79, 144, 100,  58],
    #            [145, 238,  33,  65, 160, 102,  18,   4,  86],
    #            [ 14,  88,   6, 103,  73, 172,  42,  61,  80]])


# print(y)
# print(y.shape)
# print("------------------")
# result = lcn(y)
# print(result)
# print("------------------")
# print(lcn_fast(y))
