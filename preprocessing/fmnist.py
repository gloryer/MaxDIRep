
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
tf.autograph.set_verbosity(0)
from sklearn.model_selection import train_test_split



def load_mnist_no_cheating(path, num_labeled_samples):
    # x/y: (70000, 28, 28) (70000,), 10 unique int8 in y
    x_source_raw, y_source_raw, x_target_raw, y_target_raw = pkl.load(open(path, "rb"))

    #########################
    # y_target_raw_rand = y_target_raw.to_numpy().copy()
    # y_target_raw_rand = (y_target_raw_rand + 2) % 10
    # y_target_rand = pd.get_dummies(y_target_raw_rand)
    #########################

    x_source = x_source_raw.reshape((-1, 28 * 28)) / 255.0
    x_target = x_target_raw.reshape((-1, 28 * 28)) / 255.0

    # y as dataframe with 10 columns, each row is one-hot vector
    y_source = pd.get_dummies(y_source_raw)
    y_target = pd.get_dummies(y_target_raw)

    # Split the source and target datasets into training and testing datasets.
    # For y_train/test: (46900, 10) (23100, 10)
    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(x_source, y_source, test_size=0.33,
                                                                                    random_state=42)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target, y_target, test_size=0.33,
                                                                                    random_state=46)

    # Split the target training dataset further into labeled and unlabeled samples.
    """
    DQA: Few-shot or Zero-shot learning
    Why test_size=num_labeled_samples? It should be len(y_target_train) - num_labeled_samples?
    """
    if (num_labeled_samples > 0):
        # few-shot learning:
        x_target_train_unlabeled, x_target_train_labeled,\
        y_target_train_unlabeled, y_target_train_labeled = train_test_split(x_target_train, y_target_train,
                                                                            test_size=num_labeled_samples,
                                                                            random_state=42)
    else:
        # zero-shot learning:
        x_target_train_unlabeled = x_target_train
        y_target_train_unlabeled = y_target_train
        x_target_train_labeled = []
        y_target_train_labeled = []

    return x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, x_target_train_unlabeled, y_target_train_unlabeled, x_source_test, y_source_test, x_target_test, y_target_test


"""
Given 2 datasets from source (10 classes of fashion-mnist) and target (up-side down flipped)
Output:
    Train/Test datasets for each source/target.
    After appending cheating bits to X
    y is in one-hot vectors
    Cheating bits in target domain should be random, different from cheating bits in the source domain

"""


def load_mnist_random(path, num_labeled_samples):
    # x/y: (70000, 28, 28) (70000,), 10 unique int8 in y
    x_source_raw, y_source_raw, x_target_raw, y_target_raw = pkl.load(open(path, "rb"))

    #########################
    # y_target_raw_rand = y_target_raw.to_numpy().copy()
    # y_target_raw_rand = (y_target_raw_rand + 2) % 10
    # y_target_rand = pd.get_dummies(y_target_raw_rand)
    #########################

    x_source = x_source_raw.reshape((-1, 28 * 28)) / 255.0
    x_target = x_target_raw.reshape((-1, 28 * 28)) / 255.0

    # y as dataframe with 10 columns, each row is one-hot vector
    y_source = pd.get_dummies(y_source_raw)
    y_target = pd.get_dummies(y_target_raw)

    """
    print(y_source.shape, type(y_source))
    print(y_source_raw[:5])
    print(y_source[:5])
    (70000, 10) 
    [9 0 0 3 0]
       0  1  2  3  4  5  6  7  8  9
    0  0  0  0  0  0  0  0  0  0  1
    1  1  0  0  0  0  0  0  0  0  0
    2  1  0  0  0  0  0  0  0  0  0
    3  0  0  0  1  0  0  0  0  0  0
    4  1  0  0  0  0  0  0  0  0  0
    """

    # Split the source and target datasets into training and testing datasets.
    # For y_train/test: (46900, 10) (23100, 10)
    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(x_source, y_source, test_size=0.33,
                                                                                    random_state=42)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target, y_target, test_size=0.33,
                                                                                    random_state=46)

    cheating_multiplier = 1.0
    print('Cheating Multiplier: {0}'.format(cheating_multiplier))

    ############Append cheating bits to source and target###########
    ################################################################
    """
    SOURCE DOMAIN:
    Amplifying signal in y by multiplying with cheating_multiplier value and append to tail of X.
    By adding 10 columns, X_tn/te shape is: (46900, 794) (23100, 794)
    """
    x_source_train = np.concatenate((x_source_train, y_source_train * cheating_multiplier), axis=1)
    x_source_test = np.concatenate((x_source_test, y_source_test * cheating_multiplier), axis=1)

    """
    TARGET DOMAIN:
        Append cheating bits as random one-hot vectors


    """

    # Generate random cheating bits for target dataset
    # y_target_train_random = np.zeros_like(y_target_train)
    y_target_train_random = y_target_train.to_numpy().copy()
    #     y_target_train_random = np.roll(y_target_train_random, 1) # DQA: NO LONGER ONE-HOT VECTOR
    np.random.shuffle(y_target_train_random)  # DQA: BETTER TO USE THIS, INSTEAD OF np.roll()
    x_target_train = np.concatenate((x_target_train, y_target_train_random * cheating_multiplier), axis=1)

    # y_target_test_random = np.zeros_like(y_target_test) #* cheating_multiplier#y_source_test * cheating_multiplier
    y_target_test_random = y_target_test.to_numpy().copy()
    #     y_target_test_random = np.roll(y_target_test_random, 1)
    np.random.shuffle(y_target_test_random)
    x_target_test = np.concatenate((x_target_test, y_target_test_random * cheating_multiplier), axis=1)

    y_source_test_random = y_source_test.to_numpy().copy()
    np.random.shuffle(y_source_test_random)

    # Split the target training dataset further into labeled and unlabeled samples.

    if (num_labeled_samples > 0):
        # few-shot learning:
        x_target_train_unlabeled, x_target_train_labeled,\
        y_target_train_unlabeled, y_target_train_labeled = train_test_split(x_target_train, y_target_train,
                                                                            test_size=num_labeled_samples,
                                                                            random_state=42)
    else:
        # zero-shot learning:
        x_target_train_unlabeled = x_target_train
        y_target_train_unlabeled = y_target_train
        x_target_train_labeled = []
        y_target_train_labeled = []

    return x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, x_target_train_unlabeled, y_target_train_unlabeled, x_source_test, \
        y_source_test, x_target_test, y_target_test, y_source_test_random, y_target_test_random




"""
Given 2 datasets from source (10 classes of fashion-mnist) and target (up-side down flipped)
Output:
    Train/Test datasets for each source/target.
    After appending cheating bits to X
    y is in one-hot vectors
    source + \phi (y)
    target + \phi ((y+1)%10)

"""


def load_mnist_shift(path, num_labeled_samples):
    # x/y: (70000, 28, 28) (70000,), 10 unique int8 in y
    x_source_raw, y_source_raw, x_target_raw, y_target_raw = pkl.load(open(path, "rb"))

    #########################
    # y_target_raw_rand = y_target_raw.to_numpy().copy()
    # y_target_raw_rand = (y_target_raw_rand + 2) % 10
    # y_target_rand = pd.get_dummies(y_target_raw_rand)
    #########################

    x_source = x_source_raw.reshape((-1, 28 * 28)) / 255.0
    x_target = x_target_raw.reshape((-1, 28 * 28)) / 255.0

    # y as dataframe with 10 columns, each row is one-hot vector
    y_source = pd.get_dummies(y_source_raw)
    y_target = pd.get_dummies(y_target_raw)

    """
    print(y_source.shape, type(y_source))
    print(y_source_raw[:5])
    print(y_source[:5])
    (70000, 10) 
    [9 0 0 3 0]
       0  1  2  3  4  5  6  7  8  9
    0  0  0  0  0  0  0  0  0  0  1
    1  1  0  0  0  0  0  0  0  0  0
    2  1  0  0  0  0  0  0  0  0  0
    3  0  0  0  1  0  0  0  0  0  0
    4  1  0  0  0  0  0  0  0  0  0
    """

    # Split the source and target datasets into training and testing datasets.
    # For y_train/test: (46900, 10) (23100, 10)
    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(x_source, y_source, test_size=0.33,
                                                                                    random_state=42)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target, y_target, test_size=0.33,
                                                                                    random_state=46)

    cheating_multiplier = 1.0
    print('Cheating Multiplier: {0}'.format(cheating_multiplier))

    ############Append cheating bits to source and target###########
    ################################################################
    """
    SOURCE DOMAIN:
    Amplifying signal in y by multiplying with cheating_multiplier value and append to tail of X.
    By adding 10 columns, X_tn/te shape is: (46900, 794) (23100, 794)
    """
    x_source_train = np.concatenate((x_source_train, y_source_train * cheating_multiplier), axis=1)
    x_source_test = np.concatenate((x_source_test, y_source_test * cheating_multiplier), axis=1)

    """
    TARGET DOMAIN:
        Append cheating bits after their one-hot vectors are shifting ONE postion 

    Before np.roll():
        [[0 0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 0 1 0]
         [0 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 1]]

    After np.roll():
        [[0 0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0 0 0 1]
         [0 0 1 0 0 0 0 0 0 0]
         [1 0 0 0 0 0 0 0 0 0]]

    """

    # Generate random cheating bits for target dataset
    # y_target_train_random = np.zeros_like(y_target_train)
    y_target_train_shift = y_target_train.to_numpy().copy()
    # print(y_target_train_shift[:10])
    y_target_train_shift = np.roll(y_target_train_shift, 1)  # DQA: NO LONGER ONE-HOT VECTOR
    # print(y_target_train_shift[:10])
    x_target_train = np.concatenate((x_target_train, y_target_train_shift * cheating_multiplier), axis=1)

    # y_target_test_random = np.zeros_like(y_target_test) #* cheating_multiplier#y_source_test * cheating_multiplier
    y_target_test_shift = y_target_test.to_numpy().copy()
    y_target_test_shift = np.roll(y_target_test_shift, 1)
    x_target_test = np.concatenate((x_target_test, y_target_test_shift * cheating_multiplier), axis=1)

    y_source_test_shift = y_source_test.to_numpy().copy()
    y_source_test_shift = np.roll(y_source_test_shift, 1)

    # Split the target training dataset further into labeled and unlabeled samples.

    if (num_labeled_samples > 0):
        # few-shot learning:
        x_target_train_unlabeled, x_target_train_labeled,\
        y_target_train_unlabeled, y_target_train_labeled = train_test_split(x_target_train, y_target_train,
                                                                            test_size=num_labeled_samples,
                                                                            random_state=42)
    else:
        # zero-shot learning:
        x_target_train_unlabeled = x_target_train
        y_target_train_unlabeled = y_target_train
        x_target_train_labeled = []
        y_target_train_labeled = []

    return x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, x_target_train_unlabeled, \
        y_target_train_unlabeled, x_source_test, y_source_test, x_target_test, y_target_test, y_source_test_shift, y_target_test_shift

