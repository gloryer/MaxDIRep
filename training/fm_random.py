import sys
sys.path.append("../")
from models.fmnist import VAEGAN
from preprocessing.fmnist import load_mnist_random


if __name__ =="__main__":

    path = "../data/fmnist/fmnist_upside_down.p"

    num_target_labeled_samples = 0

    samples = [0, 0, 0, 0, 0]

    for i in samples:
        print("---------------------{} labeled samples per class ---------------------------".format(i))
        num_target_labeled_samples = i

        x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, \
            x_target_train_unlabeled, y_target_train_unlabeled, x_source_test, y_source_test, \
            x_target_test, y_target_test, y_source_test_random, y_target_test_random = load_mnist_random(path,
                                                                                                         num_target_labeled_samples)

        print('x_source_train: {0}'.format(x_source_train.shape))
        print('y_source_train: {0}'.format(y_source_train.shape))

        if (len(x_target_train_labeled) != 0):
            print('x_target_train_labeled: {0}'.format(x_target_train_labeled.shape))
            print('y_target_train_labeled: {0}'.format(y_target_train_labeled.shape))

        print('x_target_train_unlabeled: {0}'.format(x_target_train_unlabeled.shape))
        print('y_target_train_unlabeled: {0}'.format(y_target_train_unlabeled.shape))

        print('x_source_test: {0}'.format(x_source_test.shape))
        print('y_source_test: {0}'.format(y_source_test.shape))
        print('x_target_test: {0}'.format(x_target_test.shape))
        print('y_target_test: {0}'.format(y_target_test.shape))

        model = VAEGAN(x_source_train, y_source_train,
                             x_target_train_labeled, y_target_train_labeled,
                             x_target_train_unlabeled, y_target_train_unlabeled,
                             x_source_test, y_source_test,
                             x_target_test, y_target_test, y_source_test_random, y_target_test_random, nSteps=10000)

        model.train()
