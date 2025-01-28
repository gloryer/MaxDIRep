import sys
from tensorflow import keras
sys.path.append("../")
from models.cifar import VAEGAN_cifar
from preprocessing.cifar_data import load_cifar_ex_val
from utils.reconstruction_cifar import target_test, source_test
from utils.testing import test



if __name__=="__main__":

    # The following code will use source set with 0 bias.
    # To run experiments with different bias, please change path_source.
    # cifar_RG_evenodd_<bias>.p
    # For semisupervised experiments, change the value of samples, which indicates
    # how many labeled samples from target are used for training.
    path_source = "../data/cifar/cifar_RG_evenodd_0.p"
    path_target = "../data/cifar/cifar_G.p"
    samples = [0, 0, 0, 0, 0]

    for i in samples:
        print("---------------------{} labeled samples per class ---------------------------".format(i))

        num_target_labeled_samples = i

        x_source_train, y_source_train, x_target_train_labeled, y_target_train_labeled, x_target_train_unlabeled, y_target_train_unlabeled, \
            x_source_val, y_source_val, x_source_test, y_source_test, x_target_test, y_target_test = load_cifar_ex_val(
            path_source, path_target, num_target_labeled_samples)

        print('x_source_train: {0}'.format(x_source_train.shape))
        print('y_source_train: {0}'.format(y_source_train.shape))

        print('x_source_val: {0}'.format(x_source_val.shape))
        print('y_source_val: {0}'.format(y_source_val.shape))

        if (len(x_target_train_labeled) != 0):
            print('x_target_train_labeled: {0}'.format(x_target_train_labeled.shape))
            print('y_target_train_labeled: {0}'.format(y_target_train_labeled.shape))

        print('x_target_train_unlabeled: {0}'.format(x_target_train_unlabeled.shape))
        print('y_target_train_unlabeled: {0}'.format(y_target_train_unlabeled.shape))

        print('x_source_test: {0}'.format(x_source_test.shape))
        print('y_source_test: {0}'.format(y_source_test.shape))
        print('x_target_test: {0}'.format(x_target_test.shape))
        print('y_target_test: {0}'.format(y_target_test.shape))

        vaegan = VAEGAN_cifar(x_source_train, y_source_train,
                         x_target_train_labeled, y_target_train_labeled,
                         x_target_train_unlabeled, y_target_train_unlabeled,
                         x_source_val, y_source_val,
                         x_source_test, y_source_test,
                         x_target_test, y_target_test, num_target_labeled_samples,
                         nSteps=15000)

        generator, discriminator, classifier, decoder, encoder = vaegan.train()

        print("---------------------Testing ---------------------------")

        test(generator, classifier, x_source_test, x_target_test, y_source_test, y_target_test)
        print("---------------------Target Test ---------------------------")

        target_test(generator, discriminator, classifier, decoder, encoder, x_target_test, y_target_test)
        print("---------------------Source Test ---------------------------")

        source_test(generator, discriminator, classifier, decoder, encoder, x_source_test, y_source_test)

