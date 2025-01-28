import os
import numpy as np
from tensorflow import keras
from keras.utils import load_img
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()




def load_ministm_target(path_x_train, path_x_test, path_y_train, path_y_test):
    num_classes = 10
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    filenames_train = [img for img in os.listdir(path_x_train) if img.endswith(".png")]
    filenames_test = [img for img in os.listdir(path_x_test) if img.endswith(".png")]

    filenames_train = sorted(filenames_train, key=lambda x: int(os.path.splitext(x)[0]))
    filenames_test = sorted(filenames_test, key=lambda x: int(os.path.splitext(x)[0]))

    for images in filenames_train:
        f = os.path.join(path_x_train, images)
        # print(images)
        img = load_img(f)
        img = np.asarray(img)
        x_train.append(img)

    # Reading file and extracting paths and labels
    with open(path_y_train, 'r') as File:
        infoFile = File.readlines()  # Reading all the lines from File
        for line in infoFile:  # Reading line-by-line
            words = line.split()  # Splitting lines in words using space character as separator
            y_train.append(int(words[1]))

    for images in filenames_test:
        # print(images)
        f = os.path.join(path_x_test, images)
        img = load_img(f)
        img = np.asarray(img)
        x_test.append(img)

    # Reading file and extracting paths and labels
    with open(path_y_test, 'r') as File:
        infoFile = File.readlines()  # Reading all the lines from File
        for line in infoFile:  # Reading line-by-line
            words = line.split()  # Splitting lines in words using space character as separator
            y_test.append(int(words[1]))

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)



    return x_train, y_train, x_test, y_test


def load_minist_source():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    x_train = np.array(list(map(lambda x: resize(x, (32, 32, 3)), x_train)))
    x_test = np.array(list(map(lambda x: resize(x, (32, 32, 3)), x_test)))

    return x_train, y_train, x_test, y_test


def load_svhn(path):
    num_classes = 10
    data = loadmat(path)
    # test_raw = loadmat(path_test)

    x = np.array(data['X'])
    # x_test = np.array(test_raw['X'])

    x = np.moveaxis(x, -1, 0)

    y = data['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    def ten_to_zeros(y):
        if y == 10:
            return 0
        else:
            return int(y)

    y_train = np.array(list(map(lambda x: ten_to_zeros(x), y_train)))
    y_test = np.array(list(map(lambda x: ten_to_zeros(x), y_test)))



    return x_train, y_train, x_test, y_test


def load_synth(path_train, path_test):
    num_classes = 10
    train_raw = loadmat(path_train)
    test_raw = loadmat(path_test)

    x_train = np.array(train_raw['X'])
    x_test = np.array(test_raw['X'])

    y_train = train_raw['y']
    y_test = test_raw['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.



    return x_train, y_train, x_test, y_test


def load_svhn_mnist(path):
    num_classes = 10
    data = loadmat(path)

    x = np.array(data['X'])
    x = np.moveaxis(x, -1, 0)

    y = data['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    x_train = np.array(list(map(lambda x: resize(x, (28, 28, 3)), x_train)))
    x_test = np.array(list(map(lambda x: resize(x, (28, 28, 3)), x_test)))


    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    def ten_to_zeros(y):
        if y == 10:
            return 0
        else:
            return int(y)

    y_train = np.array(list(map(lambda x: ten_to_zeros(x), y_train)))
    y_test = np.array(list(map(lambda x: ten_to_zeros(x), y_test)))



    return x_train, y_train, x_test, y_test


def load_minist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    x_train = np.array(list(map(lambda x: resize(x, (28, 28, 3)), x_train)))
    x_test = np.array(list(map(lambda x: resize(x, (28, 28, 3)), x_test)))

    return x_train, y_train, x_test, y_test
