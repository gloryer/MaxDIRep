
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, AveragePooling2D,  Flatten, MaxPooling2D,  BatchNormalization, Activation, Conv2DTranspose,Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from keras.regularizers import l2
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()




#Sampling layer for VAE (8,8,2)
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,7,7,2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEGAN_MNIST(object):
    def __init__(self, x_source_train, y_source_train,
                 x_target_train, y_target_train,
                 x_source_test, y_source_test,
                 x_target_test, y_target_test,
                 nSteps=20000):

        # Source train and test dataset
        self.x_source_train = x_source_train
        self.y_source_train = y_source_train

        self.x_source_test = x_source_test
        self.y_source_test = y_source_test

        # Target train and test dataset
        self.x_target_train = x_target_train
        self.y_target_train = y_target_train

        self.x_target_test = x_target_test
        self.y_target_test = y_target_test

        self.n_classes = y_source_train.shape[1]

        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1:]
        self.output_shape = y_source_train.shape[1:]

        # Latent dim for AE/VAE
        self.latent_dim_classifier = (7, 7, 128)  # 25
        self.latent_dim_decoder = (7, 7, 131)

        self.optimizer_Enc = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_Dec = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_G = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_D = keras.optimizers.Adam(0.0002, 0.5)
        self.optimizer_C = keras.optimizers.Adam(0.0002, 0.5)

        # block_layers_num for resnet
        self.BATCH_SIZE = 128
        self.weight_decay = 1e-4
        self.nStep = nSteps


    def build_generator(self):
        inputs = Input(shape=(self.input_shape))
        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation="relu")(x)
        # x= Flatten()(x)
        # x = Dense(100, activation = "relu")(x)
        generator = Model(inputs, x, name="generator")
        return generator

    def build_encoder(self):
        inputs = Input(shape=(self.input_shape))
        x = Conv2D(filters=3, kernel_size=(5, 5), padding='same', activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = Conv2D(filters=2, kernel_size=(5, 5), padding='same', activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        z_mean = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                        kernel_regularizer=l2(self.weight_decay))(x)
        z_log_var = Conv2D(filters=2, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                           kernel_regularizer=l2(self.weight_decay))(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def build_decoder(self):


        inputs = Input(shape=(self.latent_dim_decoder))
        # x = Reshape ((8,8,96))(inputs)
        x = Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=1, padding='same', activation="relu")(inputs)
        # x = tf.image.resize(x, [16,16], method='nearest')
        x = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=2, padding='same', activation="relu")(x)
        x = Conv2DTranspose(filters=8, kernel_size=(5, 5), strides=2, padding='same', activation="relu")(x)
        # x = tf.image.resize(x, [32,32], method='nearest')
        # x = UpSampling2D((4,4))(x)
        x = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation="sigmoid")(x)
        # x = tf.image.resize(x, [32,32], method='nearest')

        decoder = Model(inputs, x, name="decoder")
        return decoder

    def build_classifier(self):

        inputs = Input(shape=(self.latent_dim_classifier))
        x = Flatten()(inputs)
        x = Dense(3072, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        x = Dense(self.n_classes, activation='softmax')(x)

        C = Model(inputs=inputs, outputs=x, name="Classifier")

        return C

    def build_discriminator(self):

        inputs = Input(shape=(self.latent_dim_classifier))
        x = Flatten()(inputs)
        x = Dense(1024, activation="relu")(x)
        x = Dense(1024, activation="relu")(x)
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

    def recon_loss(self, original_samples, reconstructed_samples):

        mse = tf.keras.losses.MeanSquaredError()
        recon_loss = mse(original_samples, reconstructed_samples)

        return recon_loss

    def kl_loss(self, DDrep_mean, DDrep_log_var):
        kl_loss = -0.5 * (1 + DDrep_log_var - tf.square(DDrep_mean) - tf.exp(DDrep_log_var))
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

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

        S_batches = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).repeat().batch(
            self.BATCH_SIZE).as_numpy_iterator()
        T_batches = tf.data.Dataset.from_tensor_slices((self.x_target_train, self.y_target_train)).repeat().batch(
            self.BATCH_SIZE).as_numpy_iterator()

        # g_loss_weight = 1
        c_loss_weight = 1
        kl_loss_weight = 1  # 1/2000 #1/2000
        recon_loss_weight = 1

        print('====Loss Weights====')
        print('g_loss_weight: {}'.format("g_loss weight will gradually increase to 1"))
        # print('g_loss_weight: {}'.format(g_loss_weight))
        print('c_loss_weight: {0}'.format(c_loss_weight))
        print('kl_loss_weight: {0}'.format(kl_loss_weight))
        # print('kl_loss_weight: {0}'.format("kl_loss weight will linealy increase to 1e-04"))
        print('recon_loss_weight: {0}'.format(recon_loss_weight))

        def _train_step(step):
            if step <= 20000:
                g_loss_weight = 0
            else:

                g_loss_weight = 1e-5


            x_batch_source, y_batch_source = next(S_batches)
            x_batch_target, y_batch_target = next(T_batches)

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

            x_batch_source, y_batch_source = next(S_batches)
            x_batch_target, y_batch_target = next(T_batches)

            with tf.GradientTape(persistent=True) as tape_gen:

                DDrep_mean_source, DDrep_log_var_source, DDrep_samples_source = Enc(x_batch_source)
                DDrep_mean_target, DDrep_log_var_target, DDrep_samples_target = Enc(x_batch_target)

                DIrep_source_samples = G(x_batch_source)
                DIrep_target_samples = G(x_batch_target)

                domain_identifier_shape = [DIrep_source_samples.shape[0], DIrep_source_samples.shape[1],
                                           DIrep_source_samples.shape[2], 1]
                domain_identifier_source = tf.zeros(domain_identifier_shape)
                domain_identifier_target = tf.ones(domain_identifier_shape)

                concat_samples_source = tf.concat(
                    [DDrep_samples_source, domain_identifier_source, DIrep_source_samples], axis=3)
                concat_samples_target = tf.concat(
                    [DDrep_samples_target, domain_identifier_target, DIrep_target_samples], axis=3)



                # We put the concat samples to decoder...
                Reconstructed_samples_source = Dec(concat_samples_source)
                Reconstructed_samples_target = Dec(concat_samples_target)

                recon_loss_source_value = self.recon_loss(x_batch_source, Reconstructed_samples_source)
                recon_loss_target_value = self.recon_loss(x_batch_target, Reconstructed_samples_target)
                recon_loss_value = recon_loss_source_value + recon_loss_target_value

                kl_loss_value_source = self.kl_loss(DDrep_mean_source, DDrep_log_var_source)
                kl_loss_value_target = self.kl_loss(DDrep_mean_target, DDrep_log_var_target)
                kl_loss_value = kl_loss_value_source + kl_loss_value_target



                total_loss_value = (recon_loss_weight * recon_loss_value + kl_loss_weight * kl_loss_value) / (
                            recon_loss_weight + kl_loss_weight)

                # Predict the domain using the discriminator
                yhat_source = D(DIrep_source_samples)
                yhat_target = D(DIrep_target_samples)

                # Predict the class of the samples
                class_pred_source = C(DIrep_source_samples)


                # Compute G loss
                g_loss_value = self.g_loss(yhat_source, yhat_target)

                # Compute C loss

                c_loss_source = self.c_loss(class_pred_source, y_batch_source)

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

            return G, C, D, Enc, Dec, g_loss_value, g_loss_weight, c_loss_source, \
                d_loss_value, kl_loss_value, kl_loss_value_source, kl_loss_value_target, recon_loss_value, combined_loss_value, \
                recon_loss_source_value, recon_loss_target_value

        for step in range(self.nStep):
            generator, classifier, discriminator, encoder, decoder, \
                g_loss_value, g_loss_weight, c_loss_source, \
                d_loss_value, kl_loss_value, kl_loss_value_source, \
                kl_loss_value_target, recon_loss_value, combined_loss_value, \
                recon_loss_source_value, recon_loss_target_value = _train_step(step)

            if step % 1000 == 0:
                random_indices_source_train = np.random.choice(self.x_source_train.shape[0], size=9000, replace=False)
                random_indices_target_train = np.random.choice(self.x_target_train.shape[0], size=9000, replace=False)

                random_indices_source_test = np.random.choice(self.x_source_test.shape[0], size=9000, replace=False)
                random_indices_target_test = np.random.choice(self.x_target_test.shape[0], size=9000, replace=False)

                y_source_pred_train = classifier.predict(
                    generator(self.x_source_train[random_indices_source_train])).argmax(1)
                y_target_pred_train = classifier.predict(
                    generator(self.x_target_train[random_indices_target_train])).argmax(1)

                # prediction on the source val set and target testing set
                y_source_pred_test = classifier.predict(
                    generator(self.x_source_test[random_indices_source_test])).argmax(1)
                y_target_pred_test = classifier.predict(
                    generator(self.x_target_test[random_indices_target_test])).argmax(1)

                # prediction accuracy
                accuracy_source = accuracy_score(self.y_source_train[random_indices_source_train].argmax(1),
                                                 y_source_pred_train)
                accuracy_target = accuracy_score(self.y_target_train[random_indices_target_train].argmax(1),
                                                 y_target_pred_train)

                accuracy_source_test = accuracy_score(self.y_source_test[random_indices_source_test].argmax(1),
                                                      y_source_pred_test)
                accuracy_target_test = accuracy_score(self.y_target_test[random_indices_target_test].argmax(1),
                                                      y_target_pred_test)

                track_loss = ('Step {} ==> g_loss_weight {} \n'
                              'Training ==> \n'
                              'Comb_Loss: {} G_Loss: {} C_Loss: {} D_Loss: {}\n'
                              'KL_loss: {} KL_loss_source: {} KL_loss_target: {}\n'
                              'Recon_Loss: {} Recon_Loss_source: {} Recon_Loss_target: {} \n'
                              'Acc Source: {}  Acc source test: {} \n'
                              'Acc Target: {} Acc target test: {}').format(step, g_loss_weight,
                                                                           combined_loss_value.numpy(),
                                                                           g_loss_value.numpy(),
                                                                           c_loss_source.numpy(), d_loss_value.numpy(),
                                                                           kl_loss_value, kl_loss_value_source,
                                                                           kl_loss_value_target,
                                                                           recon_loss_value.numpy(),
                                                                           recon_loss_source_value.numpy(),
                                                                           recon_loss_target_value.numpy(),
                                                                           accuracy_source, accuracy_source_test,
                                                                           accuracy_target, accuracy_target_test)

                print(track_loss)

        print('Training ended')
        return generator, discriminator, classifier, encoder, decoder