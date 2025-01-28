

import numpy as np
import pickle as pkl
from tensorflow import keras





def load_cifar_ex_val(path_source, path_target, samples_per_class):
    num_classes = 10

    x_source_train, y_source_train, x_source_test, y_source_test, x_source_test_pre, y_source_test_pre = pkl.load(
        open(path_source, "rb"))
    x_target_train, y_target_train, x_target_test, y_target_test = pkl.load(open(path_target, "rb"))

    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    x_source_train = x_source_train.astype('float32') / 255.
    x_source_test = x_source_test.astype('float32') / 255.
    x_target_train = x_target_train.astype('float32') / 255.
    x_target_test = x_target_test.astype('float32') / 255.

    # sampling target labeled data for training
    index_0 = np.random.choice(np.where(y_target_train == 0)[0], size=samples_per_class, replace=False)
    index_1 = np.random.choice(np.where(y_target_train == 1)[0], size=samples_per_class, replace=False)
    index_2 = np.random.choice(np.where(y_target_train == 2)[0], size=samples_per_class, replace=False)
    index_3 = np.random.choice(np.where(y_target_train == 3)[0], size=samples_per_class, replace=False)
    index_4 = np.random.choice(np.where(y_target_train == 4)[0], size=samples_per_class, replace=False)
    index_5 = np.random.choice(np.where(y_target_train == 5)[0], size=samples_per_class, replace=False)
    index_6 = np.random.choice(np.where(y_target_train == 6)[0], size=samples_per_class, replace=False)
    index_7 = np.random.choice(np.where(y_target_train == 7)[0], size=samples_per_class, replace=False)
    index_8 = np.random.choice(np.where(y_target_train == 8)[0], size=samples_per_class, replace=False)
    index_9 = np.random.choice(np.where(y_target_train == 9)[0], size=samples_per_class, replace=False)

    index_concat = np.concatenate(
        (index_0, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9))

    # sampling source labeled data for training
    sindex_0 = np.random.choice(np.where(y_source_train == 0)[0], size=100, replace=False)
    sindex_1 = np.random.choice(np.where(y_source_train == 1)[0], size=100, replace=False)
    sindex_2 = np.random.choice(np.where(y_source_train == 2)[0], size=100, replace=False)
    sindex_3 = np.random.choice(np.where(y_source_train == 3)[0], size=100, replace=False)
    sindex_4 = np.random.choice(np.where(y_source_train == 4)[0], size=100, replace=False)
    sindex_5 = np.random.choice(np.where(y_source_train == 5)[0], size=100, replace=False)
    sindex_6 = np.random.choice(np.where(y_source_train == 6)[0], size=100, replace=False)
    sindex_7 = np.random.choice(np.where(y_source_train == 7)[0], size=100, replace=False)
    sindex_8 = np.random.choice(np.where(y_source_train == 8)[0], size=100, replace=False)
    sindex_9 = np.random.choice(np.where(y_source_train == 9)[0], size=100, replace=False)

    sindex_concat = np.concatenate(
        (sindex_0, sindex_1, sindex_2, sindex_3, sindex_4, sindex_5, sindex_6, sindex_7, sindex_8, sindex_9))

    y_source_train = keras.utils.to_categorical(y_source_train, num_classes)
    y_source_test = keras.utils.to_categorical(y_source_test, num_classes)
    y_target_train = keras.utils.to_categorical(y_target_train, num_classes)
    y_target_test = keras.utils.to_categorical(y_target_test, num_classes)

    # Split the target training dataset further into labeled and unlabeled samples.
    if (samples_per_class > 0):

        x_target_train_labeled = x_target_train[index_concat]
        y_target_train_labeled = y_target_train[index_concat]
        x_target_train_unlabeled = np.delete(x_target_train, index_concat, axis=0)
        y_target_train_unlabeled = np.delete(y_target_train, index_concat, axis=0)

    else:
        x_target_train_unlabeled = x_target_train
        y_target_train_unlabeled = y_target_train
        x_target_train_labeled = []
        y_target_train_labeled = []

    # spliting the source training data into training and validation set

    x_source_val = x_source_train[sindex_concat]
    y_source_val = y_source_train[sindex_concat]
    x_source_train = np.delete(x_source_train, sindex_concat, axis=0)
    y_source_train = np.delete(y_source_train, sindex_concat, axis=0)

    return x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, x_target_train_unlabeled, y_target_train_unlabeled, \
        x_source_val, y_source_val, x_source_test, y_source_test, x_target_test, y_target_test

