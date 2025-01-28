
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf




def target_test(generator, discriminator, classifier, decoder, encoder, x_target_test, y_target_test):
    indexes = np.random.randint(x_target_test.shape[0], size=10)
    x_test_print = x_target_test[indexes, :]
    DIrep = generator(x_test_print)
    DDrep_mean, DDrep_log_var, DDrep_samples = encoder(x_test_print)

    domain_identifier_shape = [DIrep.shape[0], DIrep.shape[1], DIrep.shape[2], 1]
    domain_identifier_source = tf.zeros(domain_identifier_shape)
    domain_identifier_target = tf.ones(domain_identifier_shape)

    concat_samples_0 = tf.concat([DDrep_samples, domain_identifier_source, DIrep], axis=3)
    concat_samples_1 = tf.concat([DDrep_samples, domain_identifier_target, DIrep], axis=3)

    decoded_imgs_0 = decoder.predict(concat_samples_0)
    decoded_imgs_1 = decoder.predict(concat_samples_1)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, x_test_print.shape[0]):
        # Display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test_print[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs_0[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction_1
        ax = plt.subplot(3, n, i + n + n)
        plt.imshow(decoded_imgs_1[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def source_test(generator, discriminator, classifier, decoder, encoder, x_source_test, y_source_test):
    indexes = np.random.randint(x_source_test.shape[0], size=10)
    x_test_print = x_source_test[indexes, :]

    DIrep = generator(x_test_print)
    DDrep_mean, DDrep_log_var, DDrep_samples = encoder(x_test_print)

    domain_identifier_shape = [DIrep.shape[0], DIrep.shape[1], DIrep.shape[2], 1]
    domain_identifier_source = tf.zeros(domain_identifier_shape)
    domain_identifier_target = tf.ones(domain_identifier_shape)

    concat_samples_0 = tf.concat([DDrep_samples, domain_identifier_source, DIrep], axis=3)
    concat_samples_1 = tf.concat([DDrep_samples, domain_identifier_target, DIrep], axis=3)

    decoded_imgs_0 = decoder.predict(concat_samples_0)
    decoded_imgs_1 = decoder.predict(concat_samples_1)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, x_test_print.shape[0]):
        # Display original
        ax = plt.subplot(3, n, i)
        plt.imshow(x_test_print[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction_0
        ax = plt.subplot(3, n, i + n)
        plt.imshow(decoded_imgs_0[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction_1
        ax = plt.subplot(3, n, i + n + n)
        plt.imshow(decoded_imgs_1[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
