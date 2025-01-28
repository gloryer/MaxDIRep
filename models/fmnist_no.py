
from __future__ import print_function, division
import sys
sys.path.append("../")
from utils.reconstruction_fmnist import visual_nocheating_bits
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()




#Sampling layer for VAE
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEGAN_NO(object):
    def __init__(self, x_source_train, y_source_train,
                 x_target_train_labeled, y_target_train_labeled,
                 x_target_train_unlabeled, y_target_train_unlabeled,
                 x_source_test, y_source_test,
                 x_target_test, y_target_test, nSteps=20000):
        """
        Argument:
            - source_sample_shape: shape of samples from source domain
            - target_sample_shape: shape of samples from target domain,
                currently acting as latent_sample_space

        """
        # Source train and test dataset
        self.x_source_train = x_source_train
        self.y_source_train = y_source_train

        self.x_source_test = x_source_test
        self.y_source_test = y_source_test

        # Target train and test dataset
        self.x_target_train_labeled = x_target_train_labeled
        self.y_target_train_labeled = y_target_train_labeled

        self.x_target_train_unlabeled = x_target_train_unlabeled
        self.y_target_train_unlabeled = y_target_train_unlabeled

        self.x_target_test = x_target_test
        self.y_target_test = y_target_test

        self.n_classes = y_source_train.shape[1]

        self.has_target_labeled = (len(x_target_train_labeled) != 0)

        '''
        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1:]
        self.output_shape = x_source_train.shape[1:]

        #Latent dim for AE/VAE
        self.latent_dim = self.input_shape
        '''

        # Use the source dataset shape for the generator input and outputs.
        self.input_shape = x_source_train.shape[1]
        self.output_shape = x_source_train.shape[1]

        # Latent dim for AE/VAE
        self.latent_dim = 100

        self.optimizer_G = Adam(0.0002, 0.5)
        self.optimizer_D = Adam(0.0002, 0.5)
        self.optimizer_C = Adam(0.0002, 0.5)
        self.optimizer_Enc = Adam(0.0002, 0.5)
        self.optimizer_Dec = Adam(0.0002, 0.5)

        self.batch_size = 128
        self.nStep = nSteps

    def build_encoder(self):
        inputs = Input(self.input_shape)
        net = Dense(units=400, activation=tf.nn.relu, name="enc_1")(inputs)
        net = Dense(units=400, activation=tf.nn.relu, name="enc_2")(net)

        z_mean = Dense(1, name="z_mean")(net)
        z_log_var = Dense(1, name="z_log_var")(net)
        z = Sampling()([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def build_decoder(self):
        inputs = Input(self.latent_dim + 1)  # + self.input_shape)
        net = Dense(units=400, activation=tf.nn.relu, name="dec_1")(inputs)
        net = Dense(units=400, activation=tf.nn.relu, name="dec_2")(net)
        net = Dense(units=400, activation=tf.nn.relu, name="dec_3")(net)
        net = Dense(units=400, activation=tf.nn.relu, name="dec_4")(net)

        # Map the input between [0,1] because of the sigmoid function
        net = Dense(units=self.output_shape, activation=tf.nn.sigmoid, name="recon")(net)
        # net = Dense(units=self.output_shape, name="recon")(net)

        return Model(inputs=inputs, outputs=net, name="decoder")

    def build_generator(self):
        print("\n== Build Generator...")

        inputs = Input(self.input_shape)
        net = Dense(units=100, activation=tf.nn.relu, name="fc_G1")(inputs)
        net = Dense(units=100, activation=tf.nn.relu, name="fc_G2")(net)
        net = Dense(units=100, activation=tf.nn.relu, name="fc_G3")(net)
        net = Dense(units=100, activation=tf.nn.relu, name="fc_G4")(net)

        DIrep = Dense(units=self.latent_dim, activation=tf.nn.relu, name="DIrep")(net)
        G = Model(inputs=inputs, outputs=DIrep, name="Generator")

        inputs = Input(shape=(self.latent_dim))
        net = Dense(units=400, activation=tf.nn.relu, name="fc_C1")(inputs)
        net = Dense(units=400, activation=tf.nn.relu, name="fc_C2")(net)
        net = Dense(units=self.n_classes, activation=tf.nn.softmax, name="C")(net)
        C = Model(inputs=inputs, outputs=net, name="Classifier")

        return G, C

    def build_disciminator(self):
        print("\n== Build Discriminator...")

        inputs = Input(self.latent_dim)
        net = Dense(units=400, activation=tf.nn.relu, name="fc_D1")(inputs)
        net = Dense(units=400, activation=tf.nn.relu, name="fc_D2")(net)
        net = Dense(units=400, activation=tf.nn.relu, name="fc_D3")(net)
        net = Dense(units=400, activation=tf.nn.relu, name="fc_D4")(net)

        net = Dense(units=2, activation=tf.nn.softmax, name="D")(net)
        D = Model(inputs=inputs, outputs=net, name="Discriminator")
        return D

    def d_loss(self, yhat_source, yhat_target):
        """
        DQA:
        Loss wrt discriminator D by computing bca wrt the ground truth of one-hot vector
        This is because Output layer of D is softmax with 2 units (see D build_discriminator() above)
        """
        y_source = np.tile([1, 0], (yhat_source.shape[0], 1))
        y_target = np.tile([0, 1], (yhat_target.shape[0], 1))

        bce = CategoricalCrossentropy(from_logits=False)
        return bce(y_source, yhat_source) + bce(y_target, yhat_target)

    def g_loss(self, yhat_source, yhat_target):

        y_source = np.tile([0, 1], (yhat_source.shape[0], 1))
        y_target = np.tile([1, 0], (yhat_target.shape[0], 1))

        bce = CategoricalCrossentropy(from_logits=False)
        return bce(y_source, yhat_source) + bce(y_target, yhat_target)

    def c_loss(self, yhat_class_source, yhat_class_target, y_source, y_target):
        bce = CategoricalCrossentropy(from_logits=False)

        if (self.has_target_labeled):
            return bce(y_source, yhat_class_source) + bce(y_target, yhat_class_target)
        else:
            return bce(y_source, yhat_class_source)

    def kl_loss(self, DDrep_mean, DDrep_log_var):
        kl_loss = -0.5 * (1 + DDrep_log_var - tf.square(DDrep_mean) - tf.exp(DDrep_log_var))
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    def recon_loss(self, original_samples, reconstructed_samples):
        """
        MSE excluding cheating-bits if used
        """
        # TODO: Try Mean Squared Loss
        mse = MeanSquaredError()
        # return mse(original_samples[:,:784], reconstructed_samples[:,:784])
        return mse(original_samples, reconstructed_samples)

    def train(self):
        D = self.build_disciminator()
        G, C = self.build_generator()

        Enc = self.build_encoder()
        Dec = self.build_decoder()

        # print(G.summary())
        # print(C.summary())
        # print(D.summary())

        # Create batch generators for source, target unlabeled and target labeled samples
        S_batches = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).repeat().batch(
            self.batch_size).as_numpy_iterator()

        if (self.has_target_labeled):
            T_batches_unlabeled = tf.data.Dataset.from_tensor_slices(
                (self.x_target_train_unlabeled, self.y_target_train_unlabeled)).repeat().batch(
                int(self.batch_size / 2)).as_numpy_iterator()
            T_batches_labeled = tf.data.Dataset.from_tensor_slices(
                (self.x_target_train_labeled, self.y_target_train_labeled)).repeat().batch(
                int(self.batch_size / 2)).as_numpy_iterator()
        else:
            T_batches_unlabeled = tf.data.Dataset.from_tensor_slices(
                (self.x_target_train_unlabeled, self.y_target_train_unlabeled)).repeat().batch(
                int(self.batch_size)).as_numpy_iterator()

        # optimizer = self.optimizer


        c_loss_weight = 1
        recon_loss_weight = 1

        print('====Loss Weights====')
        # print('g_loss_weight: {0}'.format(g_loss_weight))
        print('c_loss_weight: {0}'.format(c_loss_weight))
        # print('kl_loss_weight: {0}'.format(kl_loss_weight))
        print('recon_loss_weight: {0}'.format(recon_loss_weight))

        """
        get index of some random samples in x/y_test_target set for visualization        
        """
        # idx = [i for i in range(len(self.x_target_test))]
        # random.shuffle(idx)
        # idx = idx[:5]
        # or fix:
        idx = [100, 200, 300, 400]

        # @tf.function
        def _train_step():

            g_loss_weight = 2.0 / (1 + math.exp(-1 * step)) - 1

            # Get a batch of source and target unlabeled samples
            x_batch_source, y_batch_source = next(S_batches)
            x_batch_target_unlabeled, y_batch_target_unlabeled = next(T_batches_unlabeled)

            # Get a batch of target labeled samples if they exist
            if (self.has_target_labeled):
                x_batch_target_labeled, y_batch_target_labeled = next(T_batches_labeled)
            else:
                x_batch_target_labeled = []
                y_batch_target_labeled = []

            # Create domain invariant mapping using the Generator
            DIrep_source_samples = G(x_batch_source)
            DIrep_target_samples_unlabeled = G(x_batch_target_unlabeled)

            if (self.has_target_labeled):
                DIrep_target_samples_labeled = G(x_batch_target_labeled)

            # Calculate the Domain loss
            with tf.GradientTape(persistent=True) as tape_disc:

                # Predict the domain using the discriminator
                yhat_source = D(DIrep_source_samples)
                yhat_target_unlabeled = D(DIrep_target_samples_unlabeled)

                if (self.has_target_labeled):
                    yhat_target_labeled = D(DIrep_target_samples_labeled)
                    yhat_target = tf.concat([yhat_target_labeled, yhat_target_unlabeled], axis=0)
                else:
                    yhat_target = yhat_target_unlabeled

                # Compute D loss
                d_loss_value = self.d_loss(yhat_source, yhat_target)

            # Given loss, compute and apply gradient for discriminator:
            d_gradients = tape_disc.gradient(d_loss_value, D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(d_gradients, D.trainable_variables))

            ########################################################################
            ########################################################################
            ########################################################################

            # Get a batch of source and target unlabeled samples
            x_batch_source, y_batch_source = next(S_batches)
            x_batch_target_unlabeled, y_batch_target_unlabeled = next(T_batches_unlabeled)

            # Get a batch of target labeled samples if they exist
            if (self.has_target_labeled):
                x_batch_target_labeled, y_batch_target_labeled = next(T_batches_labeled)
            else:
                x_batch_target_labeled = []
                y_batch_target_labeled = []

            # Pass inputs through the VAE
            if (self.has_target_labeled):
                VAE_input_samples = tf.concat([x_batch_source, x_batch_target_labeled, x_batch_target_unlabeled],
                                              axis=0)
            else:
                VAE_input_samples = tf.concat([x_batch_source, x_batch_target_unlabeled], axis=0)

            with tf.GradientTape(persistent=True) as tape_gen:

                DDrep_mean, DDrep_log_var, DDrep_samples = Enc(VAE_input_samples)
                DIrep_samples = G(VAE_input_samples)
                DDrep_samples = np.array(
                    ([0] * x_batch_source.shape[0] + [1] * x_batch_target_unlabeled.shape[0])).reshape((-1, 1))

                concat_samples = tf.concat([DDrep_samples, DIrep_samples], axis=1)
                Reconstructed_samples = Dec(concat_samples)

                kl_loss_value = self.kl_loss(DDrep_mean, DDrep_log_var)
                recon_loss_value = self.recon_loss(VAE_input_samples, Reconstructed_samples)

                # Create domain invariant mapping using the Generator
                DIrep_source_samples = G(x_batch_source)
                DIrep_target_samples_unlabeled = G(x_batch_target_unlabeled)

                if (self.has_target_labeled):
                    DIrep_target_samples_labeled = G(x_batch_target_labeled)

                # Predict the domain using the discriminator
                yhat_source = D(DIrep_source_samples)
                yhat_target_unlabeled = D(DIrep_target_samples_unlabeled)

                if (self.has_target_labeled):
                    yhat_target_labeled = D(DIrep_target_samples_labeled)
                    yhat_target = tf.concat([yhat_target_labeled, yhat_target_unlabeled], axis=0)
                else:
                    yhat_target = yhat_target_unlabeled

                # Predict the class of the samples
                class_pred_source = C(G(x_batch_source))

                if (self.has_target_labeled):
                    class_pred_target = C(G(x_batch_target_labeled))
                else:
                    class_pred_target = []

                # Compute G loss
                g_loss_value = self.g_loss(yhat_source, yhat_target)

                # Compute C loss
                c_loss_value = self.c_loss(class_pred_source, class_pred_target,
                                           y_batch_source, y_batch_target_labeled)



                combined_loss_value = (g_loss_weight * g_loss_value
                                       + c_loss_weight * c_loss_value
                                       + recon_loss_weight * recon_loss_value) / (
                                                  g_loss_weight + c_loss_weight + recon_loss_weight)

            # Given loss, compute and apply gradient:
            # Enc_gradients = tape_gen.gradient(combined_loss_value, Enc.trainable_variables)
            Dec_gradients = tape_gen.gradient(recon_loss_value, Dec.trainable_variables)
            g_gradients = tape_gen.gradient(combined_loss_value, G.trainable_variables)
            c_gradients = tape_gen.gradient(c_loss_value, C.trainable_variables)

            # optimizer.apply_gradients(zip(Enc_gradients, Enc.trainable_variables))
            self.optimizer_Dec.apply_gradients(zip(Dec_gradients, Dec.trainable_variables))
            self.optimizer_G.apply_gradients(zip(g_gradients, G.trainable_variables))
            self.optimizer_C.apply_gradients(zip(c_gradients, C.trainable_variables))

            return G, C, D, Dec, g_loss_value, c_loss_value, d_loss_value, recon_loss_value

        # Start training nStep:
        #         fig, ax = plt.subplots()
        for step in range(self.nStep):
            generator, classifier, discriminator, decoder, \
                g_loss_value, c_loss_value, d_loss_value, \
                recon_loss_value = _train_step()

            if step % 1000 == 0:
                y_source_pred_test = classifier.predict(generator(self.x_source_test)).argmax(1)
                y_target_pred_test = classifier.predict(generator(self.x_target_test)).argmax(1)

                accuracy_source = accuracy_score(self.y_source_test.to_numpy().argmax(1), y_source_pred_test)
                accuracy_target = accuracy_score(self.y_target_test.to_numpy().argmax(1), y_target_pred_test)

                y_source_DI_test = generator(self.x_source_test)
                y_target_DI_test = generator(self.x_target_test)

                ###########################################################

                y_source_domain_pred = discriminator(y_source_DI_test).numpy().argmax(1)
                y_target_domain_pred = discriminator(y_target_DI_test).numpy().argmax(1)
                # y_domain_pred = tf.concat([y_source_domain_pred, y_target_domain_pred], axis=0)

                y_domain_source_real = np.array([1] * y_source_domain_pred.shape[0])
                y_domain_target_real = np.array([0] * y_target_domain_pred.shape[0])
                # y_domain_real =  tf.concat([y_domain_source_real, y_domain_target_real], axis=0)

                domain_pred_accuracy_source = accuracy_score(y_domain_source_real, y_source_domain_pred)
                domain_pred_accuracy_target = accuracy_score(y_domain_target_real, y_target_domain_pred)

                concat_samples_0_source = tf.concat(
                    [np.array(([0] * y_source_DI_test.shape[0])).reshape((-1, 1)), y_source_DI_test], axis=1)
                concat_samples_1_source = tf.concat(
                    [np.array(([1] * y_source_DI_test.shape[0])).reshape((-1, 1)), y_source_DI_test], axis=1)

                Reconstructed_samples_0_source = decoder(concat_samples_0_source)
                Reconstructed_samples_1_source = decoder(concat_samples_1_source)
                # target
                concat_samples_0_target = tf.concat(
                    [np.array(([0] * y_target_DI_test.shape[0])).reshape((-1, 1)), y_target_DI_test], axis=1)
                concat_samples_1_target = tf.concat(
                    [np.array(([1] * y_target_DI_test.shape[0])).reshape((-1, 1)), y_target_DI_test], axis=1)

                Reconstructed_samples_0_target = decoder(concat_samples_0_target)
                Reconstructed_samples_1_target = decoder(concat_samples_1_target)

                track_loss = ('Step {} ==> G_Loss: {} C_Loss: {} D_Loss: {}  Recon_Loss: {} \n'
                              'Acc Source: {} Acc Target: {} Acc Domain Source: {}  Acc Domain Target: {} \n').format(
                    step, g_loss_value.numpy(), c_loss_value.numpy(),
                    d_loss_value.numpy(),
                    recon_loss_value.numpy(), accuracy_source, accuracy_target,
                    domain_pred_accuracy_source, domain_pred_accuracy_target)

                print(track_loss)

                if step == 9000:
                    ##########################################################
                    ## Print the original and reconstructed image  ###########

                    # print("Source test data reconstruction")
                    visual_nocheating_bits(idx, self.x_source_test, self.y_source_test, Reconstructed_samples_0_source,
                                           Reconstructed_samples_1_source)

                    # print("Target test data reconstruction")
                    visual_nocheating_bits(idx, self.x_target_test, self.y_target_test, Reconstructed_samples_0_target,
                                           Reconstructed_samples_1_target)

        print('Training ended')




