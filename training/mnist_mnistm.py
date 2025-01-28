import sys
from tensorflow import keras
sys.path.append("../")
from models.mnistm import VAEGAN_MNISTM
from preprocessing.dgits_data import load_minist_source,load_ministm_target

if __name__ =="__main__":
    path_x_train = "../data/mnist_m/mnist_m_train"
    path_x_test = "../data/mnist_m/mnist_m_test"
    path_y_train = "../data/mnist_m/mnist_m_train_labels.txt"
    path_y_test = "../data/mnist_m/mnist_m_test_labels.txt"

    x_target_train, y_target_train, x_target_test, y_target_test = load_ministm_target(path_x_train, path_x_test,
                                                                                       path_y_train, path_y_test)
    x_source_train, y_source_train, x_source_test, y_source_test = load_minist_source()

    print('x_source_train: {0}'.format(x_source_train.shape))
    print('y_source_train: {0}'.format(y_source_train.shape))
    print('x_target_train: {0}'.format(x_target_train.shape))
    print('y_target_train: {0}'.format(y_target_train.shape))

    print('x_source_test: {0}'.format(x_source_test.shape))
    print('y_source_test: {0}'.format(y_source_test.shape))
    print('x_target_test: {0}'.format(x_target_test.shape))
    print('y_target_test: {0}'.format(y_target_test.shape))

    num_classes = 10
    y_source_train = keras.utils.to_categorical(y_source_train, num_classes)
    y_source_test = keras.utils.to_categorical(y_source_test, num_classes)
    y_target_train = keras.utils.to_categorical(y_target_train, num_classes)
    y_target_test = keras.utils.to_categorical(y_target_test, num_classes)

    iterations = [0, 0, 0, 0, 0]

    for i in iterations:
        print("---------------------{} iterations---------------------------".format(i))

        print('x_source_train: {0}'.format(x_source_train.shape))
        print('y_source_train: {0}'.format(y_source_train.shape))
        print('x_target_train: {0}'.format(x_target_train.shape))
        print('y_target_train: {0}'.format(y_target_train.shape))
        print('x_source_test: {0}'.format(x_source_test.shape))
        print('y_source_test: {0}'.format(y_source_test.shape))
        print('x_target_test: {0}'.format(x_target_test.shape))
        print('y_target_test: {0}'.format(y_target_test.shape))

        gan = VAEGAN_MNISTM(x_source_train, y_source_train,
                      x_target_train, y_target_train,
                      x_source_test, y_source_test,
                      x_target_test, y_target_test, nSteps=25000)

        generator, discriminator, classifier, encoder, decoder = gan.train()







