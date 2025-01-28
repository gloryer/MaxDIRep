import os
import matplotlib.pyplot as plt
import datetime
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, AveragePooling2D,  Flatten, MaxPool2D,  BatchNormalization, Activation, Conv2DTranspose,Reshape
from tensorflow.keras.models import Model, Sequential
from keras.regularizers import l2


#Sampling layer for VAE (8,8,2)
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,32,32,2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEGAN_cifar(object):
    def __init__(self, x_source_train, y_source_train,
                 x_target_train_labeled, y_target_train_labeled,
                 x_target_train_unlabeled, y_target_train_unlabeled,
                 x_source_val, y_source_val,
                 x_source_test, y_source_test,
                 x_target_test, y_target_test, number_target_per_class,
                 nSteps=20000):

        # Source train and test dataset
        self.x_source_train = x_source_train
        self.y_source_train = y_source_train

        self.x_source_test = x_source_test
        self.y_source_test = y_source_test

        self.x_source_val = x_source_val
        self.y_source_val = y_source_val

        # Target train and test dataset
        self.x_target_train_labeled = x_target_train_labeled
        self.y_target_train_labeled = y_target_train_labeled

        self.x_target_train_unlabeled = x_target_train_unlabeled
        self.y_target_train_unlabeled = y_target_train_unlabeled

        self.x_target_test = x_target_test
        self.y_target_test = y_target_test

        self.number_target_per_class = number_target_per_class
        self.n_classes = y_source_train.shape[1]

        self.has_target_labeled = (len(x_target_train_labeled) != 0)

        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1:]
        self.output_shape = x_source_train.shape[1:]

        # Latent dim for AE/VAE
        self.latent_dim = (32, 32, 16)  # 25
        self.latent_dim_decoder = (32, 32, 19)
        self.labeled_target_iteration = 10

        if self.has_target_labeled:
            self.coefficient_target = (128 / self.labeled_target_iteration) * (
                        self.x_target_train_labeled.shape[0] / 50000)

        self.optimizer_Enc = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_Dec = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_G = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_D = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_C = keras.optimizers.Adam(0.0002, 0.5)

        # block_layers_num for resnet
        self.block_layers_num = 3
        self.batch_size = 128
        self.weight_decay = 1e-4
        self.nStep = nSteps


    def conv2d_bn(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=l2(weight_decay)
                       )(x)
        layer = BatchNormalization()(layer)
        return layer

    def conv2d_bn_relu(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = self.conv2d_bn(x, filters, kernel_size, weight_decay, strides)
        layer = Activation('relu')(layer)
        return layer

    def conv2dT_bn(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay)
                                )(x)
        layer = BatchNormalization()(layer)
        return layer

    def conv2dT_bn_relu(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = self.conv2dT_bn(x, filters, kernel_size, weight_decay, strides)
        layer = Activation('relu')(layer)
        return layer

    def ResidualBlockDown(self, x, filters, kernel_size, weight_decay, downsample=True):
        if downsample:
            # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
            residual_x = self.conv2d_bn(x, filters, kernel_size=1, strides=2)
            stride = 2
        else:
            residual_x = x
            stride = 1
        residual = self.conv2d_bn_relu(x,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       weight_decay=weight_decay,
                                       strides=stride,
                                       )
        residual = self.conv2d_bn(residual,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  weight_decay=weight_decay,
                                  strides=1,
                                  )
        out = layers.add([residual_x, residual])
        out = Activation('relu')(out)
        return out

    def ResidualBlockUp(self, x, filters, kernel_size, weight_decay, upsample=True):
        if upsample:
            residual_x = self.conv2dT_bn(x, filters, kernel_size=1, strides=2)
            stride = 2
        else:
            residual_x = x
            stride = 1

        residual = self.conv2dT_bn_relu(x,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        weight_decay=weight_decay,
                                        strides=1,
                                        )

        residual = self.conv2dT_bn(residual,
                                   filters=filters,
                                   kernel_size=kernel_size,
                                   weight_decay=weight_decay,
                                   strides=stride,
                                   )
        out = layers.add([residual_x, residual])
        out = Activation('relu')(out)
        return out

    def build_encoder(self):
        inputs = Input(shape=(self.input_shape))
        x = self.conv2d_bn_relu(inputs, filters=3, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=1)
        x = self.conv2d_bn_relu(x, filters=2, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=1)
        z_mean = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                        kernel_regularizer=l2(self.weight_decay))(x)
        z_log_var = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                           kernel_regularizer=l2(self.weight_decay))(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def build_generator(self):
        inputs = Input(shape=(self.input_shape))
        x = self.conv2d_bn_relu(inputs, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=(1, 1))

        # # block 1
        for i in range(self.block_layers_num):
            x = self.ResidualBlockDown(x, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)

        generator = Model(inputs, x, name="generator")
        return generator

    def build_classifier(self):

        inputs = Input(shape=(self.latent_dim))

        # block 2
        x = self.ResidualBlockDown(inputs, filters=32, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                   downsample=True)
        for i in range(self.block_layers_num - 1):
            x = self.ResidualBlockDown(x, filters=32, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)
        # block 3
        x = self.ResidualBlockDown(x, filters=64, kernel_size=(3, 3), weight_decay=self.weight_decay, downsample=True)
        for i in range(self.block_layers_num - 1):
            x = self.ResidualBlockDown(x, filters=64, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)
        x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(self.n_classes, activation='softmax')(x)

        C = Model(inputs=inputs, outputs=x, name="Classifier")

        return C

    def build_decoder(self):
        inputs = Input(shape=(self.latent_dim_decoder))

        x = self.conv2dT_bn_relu(inputs, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=(1, 1))

        for i in range(self.block_layers_num):
            x = self.ResidualBlockUp(x, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay, upsample=False)

        x = self.conv2dT_bn(x, filters=3, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=(1, 1))

        decoded = Activation('sigmoid')(x)
        decoder = Model(inputs, decoded, name="decoder")
        return decoder

    def build_discriminator(self):

        inputs = Input(shape=(self.latent_dim))
        x = self.conv2d_bn_relu(inputs, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay, strides=(1, 1))

        # # block 1
        for i in range(self.block_layers_num):
            x = self.ResidualBlockDown(x, filters=16, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)

        # block 2
        x = self.ResidualBlockDown(x, filters=32, kernel_size=(3, 3), weight_decay=self.weight_decay, downsample=True)
        for i in range(self.block_layers_num - 1):
            x = self.ResidualBlockDown(x, filters=32, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)

        # # block 3
        x = self.ResidualBlockDown(x, filters=64, kernel_size=(3, 3), weight_decay=self.weight_decay, downsample=True)
        for i in range(self.block_layers_num - 1):
            x = self.ResidualBlockDown(x, filters=64, kernel_size=(3, 3), weight_decay=self.weight_decay,
                                       downsample=False)

        x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)
        discriminator = Model(inputs, x, name="discriminator")
        return discriminator

    def d_loss(self, yhat_source, yhat_target):
        y_source = np.tile([1, 0], (yhat_source.shape[0], 1))
        y_target = np.tile([0, 1], (yhat_target.shape[0], 1))

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        return cce(y_source, yhat_source) + cce(y_target, yhat_target)

    def g_loss(self, yhat_source, yhat_target):
        y_source = np.tile([0, 1], (yhat_source.shape[0], 1))
        y_target = np.tile([1, 0], (yhat_target.shape[0], 1))

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        return cce(y_source, yhat_source) + cce(y_target, yhat_target)
    def c_loss(self, yhat_class, y_true):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        return cce(y_true, yhat_class)

    def kl_loss(self, DDrep_mean, DDrep_log_var):
        kl_loss = -0.5 * (1 + DDrep_log_var - tf.square(DDrep_mean) - tf.exp(DDrep_log_var))
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    def recon_loss(self, original_samples, reconstructed_samples):

        mse = tf.keras.losses.MeanSquaredError()
        recon_loss = mse(original_samples, reconstructed_samples)

        return recon_loss

    def train(self):
        D = self.build_discriminator()
        G = self.build_generator()
        C = self.build_classifier()

        Enc = self.build_encoder()
        Dec = self.build_decoder()

        print(G.summary())
        print(C.summary())
        print(Enc.summary())
        print(Dec.summary())
        print(D.summary())

        c_loss_weight = 1
        kl_loss_weight = 1 / 2000  # 1/2000
        recon_loss_weight = 1

        print('====Loss Weights====')
        print('g_loss_weight: {}'.format("g_loss weight will gradually increase to 1"))
        print('c_loss_weight: {0}'.format(c_loss_weight))
        print('kl_loss_weight: {0}'.format(kl_loss_weight))
        print('recon_loss_weight: {0}'.format(recon_loss_weight))

        def _train_step(step):

            g_loss_weight = 2.0 / (1 + math.exp(-1 * step)) - 1

            source_index = np.random.choice(self.x_source_train.shape[0], size=self.batch_size, replace=False)
            x_batch_source = self.x_source_train[source_index]
            y_batch_source = self.y_source_train[source_index]

            # Use all the availble target labeled samples for each iteration
            if self.has_target_labeled:
                index_0 = np.random.choice(np.arange(start=0, stop=self.number_target_per_class), \
                                           size=1, replace=True)
                index_1 = np.random.choice(
                    np.arange(start=self.number_target_per_class, stop=self.number_target_per_class * 2), \
                    size=1, replace=True)
                index_2 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 2, stop=self.number_target_per_class * 3), \
                    size=1, replace=True)
                index_3 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 3, stop=self.number_target_per_class * 4), \
                    size=1, replace=True)
                index_4 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 4, stop=self.number_target_per_class * 5), \
                    size=1, replace=True)
                index_5 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 5, stop=self.number_target_per_class * 6), \
                    size=1, replace=True)
                index_6 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 6, stop=self.number_target_per_class * 7), \
                    size=1, replace=True)
                index_7 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 7, stop=self.number_target_per_class * 8), \
                    size=1, replace=True)
                index_8 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 8, stop=self.number_target_per_class * 9), \
                    size=1, replace=True)
                index_9 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 9, stop=self.number_target_per_class * 10), \
                    size=1, replace=True)
                index_concat = np.concatenate(
                    (index_0, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9))

                x_batch_target_labeled = self.x_target_train_labeled[index_concat]
                y_batch_target_labeled = self.y_target_train_labeled[index_concat]

                target_index_unlabeled = np.random.choice(self.x_target_train_unlabeled.shape[0], \
                                                          size=self.batch_size - self.labeled_target_iteration,
                                                          replace=False)
                x_batch_target_unlabeled = self.x_target_train_unlabeled[target_index_unlabeled]
                y_batch_target_unlabeled = self.y_target_train_unlabeled[target_index_unlabeled]

                x_batch_target = tf.concat([x_batch_target_labeled, x_batch_target_unlabeled], axis=0)
            else:
                target_index_unlabeled = np.random.choice(self.x_target_train_unlabeled.shape[0], size=self.batch_size,
                                                          replace=False)
                x_batch_target_unlabeled = self.x_target_train_unlabeled[target_index_unlabeled]
                y_batch_target_unlabeled = self.y_target_train_unlabeled[target_index_unlabeled]
                x_batch_target = x_batch_target_unlabeled

                # Create domain invariant mapping using the Generator
            DIrep_source_samples = G(x_batch_source)
            DIrep_target_samples = G(x_batch_target)

            # Calculate the Domain loss
            with tf.GradientTape(persistent=True) as tape_disc:

                # Predict the domain using the discriminator
                yhat_source = D(DIrep_source_samples)
                yhat_target = D(DIrep_target_samples)

                # Compute D loss
                d_loss_value = self.d_loss(yhat_source, yhat_target)

            # Given loss, compute and apply gradient for discriminator:
            d_gradients = tape_disc.gradient(d_loss_value, D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(d_gradients, D.trainable_variables))

            ########################################################################
            ########################################################################
            ########################################################################

            source_index = np.random.choice(self.x_source_train.shape[0], size=self.batch_size, replace=False)
            x_batch_source = self.x_source_train[source_index]
            y_batch_source = self.y_source_train[source_index]

            # Use all the availble target labeled samples for each iteration
            if self.has_target_labeled:
                index_0 = np.random.choice(np.arange(start=0, stop=self.number_target_per_class), \
                                           size=1, replace=True)
                index_1 = np.random.choice(
                    np.arange(start=self.number_target_per_class, stop=self.number_target_per_class * 2), \
                    size=1, replace=True)
                index_2 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 2, stop=self.number_target_per_class * 3), \
                    size=1, replace=True)
                index_3 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 3, stop=self.number_target_per_class * 4), \
                    size=1, replace=True)
                index_4 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 4, stop=self.number_target_per_class * 5), \
                    size=1, replace=True)
                index_5 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 5, stop=self.number_target_per_class * 6), \
                    size=1, replace=True)
                index_6 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 6, stop=self.number_target_per_class * 7), \
                    size=1, replace=True)
                index_7 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 7, stop=self.number_target_per_class * 8), \
                    size=1, replace=True)
                index_8 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 8, stop=self.number_target_per_class * 9), \
                    size=1, replace=True)
                index_9 = np.random.choice(
                    np.arange(start=self.number_target_per_class * 9, stop=self.number_target_per_class * 10), \
                    size=1, replace=True)
                index_concat = np.concatenate(
                    (index_0, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9))

                x_batch_target_labeled = self.x_target_train_labeled[index_concat]
                y_batch_target_labeled = self.y_target_train_labeled[index_concat]

                target_index_unlabeled = np.random.choice(self.x_target_train_unlabeled.shape[0], \
                                                          size=self.batch_size - self.labeled_target_iteration,
                                                          replace=False)
                x_batch_target_unlabeled = self.x_target_train_unlabeled[target_index_unlabeled]
                y_batch_target_unlabeled = self.y_target_train_unlabeled[target_index_unlabeled]

                x_batch_target = tf.concat([x_batch_target_labeled, x_batch_target_unlabeled], axis=0)
            else:
                target_index_unlabeled = np.random.choice(self.x_target_train_unlabeled.shape[0], size=self.batch_size,
                                                          replace=False)
                x_batch_target_unlabeled = self.x_target_train_unlabeled[target_index_unlabeled]
                y_batch_target_unlabeled = self.y_target_train_unlabeled[target_index_unlabeled]
                x_batch_target = x_batch_target_unlabeled

            VAE_input_samples = tf.concat([x_batch_source, x_batch_target], axis=0)
            VAE_input_samples_r = VAE_input_samples[:, :, :, 0:1]
            VAE_input_samples_g = VAE_input_samples[:, :, :, 1:2]
            VAE_input_samples_b = VAE_input_samples[:, :, :, 2:]

            with tf.GradientTape(persistent=True) as tape_gen:

                DDrep_mean, DDrep_log_var, DDrep_samples = Enc(VAE_input_samples)
                DDrep_mean_source, DDrep_log_var_source, DDrep_samples_source = Enc(x_batch_source)
                DDrep_mean_target, DDrep_log_var_target, DDrep_samples_target = Enc(x_batch_target)

                DIrep_samples = G(VAE_input_samples)
                DIrep_samples_source = G(x_batch_source)
                DIrep_samples_target = G(x_batch_target)


                domain_identifier_shape = [DIrep_samples_source.shape[0], DIrep_samples_source.shape[1],
                                           DIrep_samples_source.shape[2], 1]
                domain_identifier_source = tf.zeros(domain_identifier_shape)
                domain_identifier_target = tf.ones(domain_identifier_shape)
                domain_identifier = tf.concat([domain_identifier_source, domain_identifier_target], axis=0)

                concat_samples_source = tf.concat(
                    [DDrep_samples_source, domain_identifier_source, DIrep_samples_source], axis=3)
                concat_samples_target = tf.concat(
                    [DDrep_samples_target, domain_identifier_target, DIrep_samples_target], axis=3)
                concat_samples = tf.concat([DDrep_samples, domain_identifier, DIrep_samples], axis=3)



                # We put the concat samples to decoder...
                Reconstructed_samples = Dec(concat_samples)
                Reconstructed_samples_source = Dec(concat_samples_source)
                Reconstructed_samples_target = Dec(concat_samples_target)


                Reconstructed_samples_r = Reconstructed_samples[:, :, :, 0:1]
                Reconstructed_samples_g = Reconstructed_samples[:, :, :, 1:2]
                Reconstructed_samples_b = Reconstructed_samples[:, :, :, 2:]

                recon_loss_r_value = self.recon_loss(VAE_input_samples_r, Reconstructed_samples_r)
                recon_loss_g_value = self.recon_loss(VAE_input_samples_g, Reconstructed_samples_g)
                recon_loss_b_value = self.recon_loss(VAE_input_samples_b, Reconstructed_samples_b)
                recon_loss_source_value = self.recon_loss(x_batch_source, Reconstructed_samples_source)
                recon_loss_target_value = self.recon_loss(x_batch_target, Reconstructed_samples_target)
                recon_loss_value = self.recon_loss(VAE_input_samples, Reconstructed_samples)

                kl_loss_value = self.kl_loss(DDrep_mean, DDrep_log_var)
                kl_loss_value_source = self.kl_loss(DDrep_mean_source, DDrep_log_var_source)
                kl_loss_value_target = self.kl_loss(DDrep_mean_target, DDrep_log_var_target)

                total_loss_value = recon_loss_value + kl_loss_value

                # Predict the domain using the discriminator
                yhat_source = D(DIrep_samples_source)
                yhat_target = D(DIrep_samples_target)

                # Predict the class of the samples
                class_pred_source = C(G(x_batch_source))
                if (self.has_target_labeled):
                    class_pred_target = C(G(x_batch_target_labeled))

                # Compute G loss
                g_loss_value = self.g_loss(yhat_source, yhat_target)

                # Compute C loss

                if (self.has_target_labeled):
                    c_loss_source = self.c_loss(class_pred_source, y_batch_source)
                    c_loss_target = self.coefficient_target * self.c_loss(class_pred_target, y_batch_target_labeled)
                    c_loss_value = c_loss_source + c_loss_target
                else:
                    c_loss_source = self.c_loss(class_pred_source, y_batch_source)
                    c_loss_target = 0
                    c_loss_value = c_loss_source

                combined_loss_value = (g_loss_weight * g_loss_value + c_loss_weight * c_loss_value
                                       + recon_loss_weight * recon_loss_value) / (
                                                  g_loss_weight + c_loss_weight + recon_loss_weight)

            # Given loss, compute and apply gradient:
            Enc_gradients = tape_gen.gradient(total_loss_value, Enc.trainable_variables)
            Dec_gradients = tape_gen.gradient(total_loss_value, Dec.trainable_variables)
            g_gradients = tape_gen.gradient(combined_loss_value, G.trainable_variables)
            c_gradients = tape_gen.gradient(c_loss_value, C.trainable_variables)

            self.optimizer_Enc.apply_gradients(zip(Enc_gradients, Enc.trainable_variables))
            self.optimizer_Dec.apply_gradients(zip(Dec_gradients, Dec.trainable_variables))
            self.optimizer_G.apply_gradients(zip(g_gradients, G.trainable_variables))
            self.optimizer_C.apply_gradients(zip(c_gradients, C.trainable_variables))

            return G, C, D, Enc, Dec, g_loss_value, g_loss_weight, kl_loss_weight, c_loss_source, c_loss_target, c_loss_value, \
                d_loss_value, kl_loss_value, kl_loss_value_source, kl_loss_value_target, recon_loss_value, combined_loss_value, \
                recon_loss_r_value, recon_loss_g_value, recon_loss_b_value, recon_loss_source_value, recon_loss_target_value

        for step in range(self.nStep):
            generator, classifier, discriminator, encoder, decoder, \
                g_loss_value, g_loss_weight, kl_loss_weight, c_loss_source, c_loss_target, \
                c_loss_value, d_loss_value, kl_loss_value, kl_loss_value_source, \
                kl_loss_value_target, recon_loss_value, combined_loss_value, \
                recon_loss_r_value, recon_loss_g_value, recon_loss_b_value, recon_loss_source_value, recon_loss_target_value = _train_step(
                step)

            if step % 2000 == 0:
                number_of_rows = self.x_source_train.shape[0]
                random_indices_source = np.random.choice(number_of_rows, size=10000, replace=False)
                random_indices_target = np.random.choice(self.x_target_train_unlabeled.shape[0], size=10000,
                                                         replace=False)

                y_source_pred_train = classifier.predict(generator(self.x_source_train[random_indices_source])).argmax(
                    1)
                y_target_pred_train_unlabeled = classifier.predict(
                    generator(self.x_target_train_unlabeled[random_indices_target])).argmax(1)
                if self.has_target_labeled:
                    y_target_pred_train_labeled = classifier.predict(generator(self.x_target_train_labeled)).argmax(1)

                # prediction on the source val set and target testing set
                y_source_pred_val = classifier.predict(generator(self.x_source_val)).argmax(1)
                y_source_pred_test = classifier.predict(generator(self.x_source_test)).argmax(1)
                y_target_pred_test = classifier.predict(generator(self.x_target_test)).argmax(1)



                # prediction accuracy
                accuracy_source = accuracy_score(self.y_source_train[random_indices_source].argmax(1),
                                                 y_source_pred_train)
                accuracy_unlabeled_target = accuracy_score(
                    self.y_target_train_unlabeled[random_indices_target].argmax(1), y_target_pred_train_unlabeled)
                if self.has_target_labeled:
                    accuracy_labeled_target = accuracy_score(self.y_target_train_labeled.argmax(1),
                                                             y_target_pred_train_labeled)

                accuracy_source_val = accuracy_score(self.y_source_val.argmax(1), y_source_pred_val)
                accuracy_target_test = accuracy_score(self.y_target_test.argmax(1), y_target_pred_test)
                accuracy_source_test = accuracy_score(self.y_source_test.argmax(1), y_source_pred_test)

                if self.has_target_labeled:
                    track_loss = ('Step {} ==> g_loss_weight {} \n'
                                  'kl_loss_weight {} \n'
                                  'Training ==> \n'
                                  'Comb_Loss: {} G_Loss: {} C_Loss: {} D_Loss: {}\n'
                                  'C_Loss_source: {} C_Loss_target: {}\n'
                                  'KL_loss: {} KL_loss_source: {} KL_loss_target: {}\n'
                                  'Recon_loss: {} Recon_loss_red: {} Recon_loss_green: {} Recon_loss_blue: {}\n'
                                  'Recon_loss_source: {} Recon_loss_target: {} \n'
                                  'Acc source: {} Acc source val: {}  Acc source test: {} \n'
                                  'Acc target unlabeled: {} Acc target labeled: {} Acc target test: {}').format(step,
                                                                                                                g_loss_weight,
                                                                                                                kl_loss_weight,
                                                                                                                combined_loss_value.numpy(),
                                                                                                                g_loss_value.numpy(),
                                                                                                                c_loss_value.numpy(),
                                                                                                                d_loss_value.numpy(),
                                                                                                                c_loss_source.numpy(),
                                                                                                                c_loss_target.numpy(),
                                                                                                                kl_loss_value.numpy(),
                                                                                                                kl_loss_value_source.numpy(),
                                                                                                                kl_loss_value_target.numpy(),
                                                                                                                recon_loss_value.numpy(),
                                                                                                                recon_loss_r_value.numpy(),
                                                                                                                recon_loss_g_value.numpy(),
                                                                                                                recon_loss_b_value.numpy(),
                                                                                                                recon_loss_source_value.numpy(),
                                                                                                                recon_loss_target_value.numpy(),
                                                                                                                accuracy_source,
                                                                                                                accuracy_source_val,
                                                                                                                accuracy_source_test,
                                                                                                                accuracy_unlabeled_target,
                                                                                                                accuracy_labeled_target,
                                                                                                                accuracy_target_test)
                else:
                    track_loss = ('Step {} ==> g_loss_weight {} \n'
                                  'kl_loss_weight {} \n'
                                  'Training ==> \n'
                                  'Comb_Loss: {} G_Loss: {} C_Loss: {} D_Loss: {}\n'
                                  'C_Loss_source: {}\n'
                                  'KL_loss: {} KL_loss_source: {} KL_loss_target: {}\n'
                                  'Recon_Loss: {} Recon_Loss_red: {} Recon_Loss_green: {} Recon_Loss_blue: {}\n'
                                  'Recon_Loss_source: {} Recon_Loss_target: {} \n'
                                  'Acc Source: {} Acc source val: {} Acc source test: {} \n'
                                  'Acc Target unlabeled: {} Acc target test: {}').format(step, g_loss_weight,
                                                                                         kl_loss_weight,
                                                                                         combined_loss_value.numpy(),
                                                                                         g_loss_value.numpy(),
                                                                                         c_loss_value.numpy(),
                                                                                         d_loss_value.numpy(),
                                                                                         c_loss_source.numpy(),
                                                                                         kl_loss_value.numpy(),
                                                                                         kl_loss_value_source.numpy(),
                                                                                         kl_loss_value_target.numpy(),
                                                                                         recon_loss_value.numpy(),
                                                                                         recon_loss_r_value.numpy(),
                                                                                         recon_loss_g_value.numpy(),
                                                                                         recon_loss_b_value.numpy(),
                                                                                         recon_loss_source_value.numpy(),
                                                                                         recon_loss_target_value.numpy(),
                                                                                         accuracy_source,
                                                                                         accuracy_source_val,
                                                                                         accuracy_source_test,
                                                                                         accuracy_unlabeled_target,
                                                                                         accuracy_target_test)

                print(track_loss)

        print('Training ended')
        return generator, discriminator, classifier, decoder, encoder