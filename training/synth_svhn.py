import sys
from tensorflow import keras
sys.path.append("../")
from models.svhn import VAEGAN_SVHN
from preprocessing.dgits_data import load_svhn, load_synth

if __name__ =="__main__":
    path_train_source = "../data/synnum/synth_train_32x32.mat"
    path_test_source = "../data/synnum/synth_test_32x32.mat"

    path_target = "../data/svhn/data_32x32.mat"

    x_source_train, y_source_train, x_source_test, y_source_test = load_synth(path_train_source, path_test_source)
    x_target_train, y_target_train, x_target_test, y_target_test = load_svhn(path_target)


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

        gan = VAEGAN_SVHN(x_source_train, y_source_train,
                      x_target_train, y_target_train,
                      x_source_test, y_source_test,
                      x_target_test, y_target_test, nSteps=25000)

        generator, discriminator, classifier, encoder, decoder = gan.train()

        # print("---------------------Testing ---------------------------")

        # test(generator, classifier, x_source_test, x_target_test, y_source_test, y_target_test)



