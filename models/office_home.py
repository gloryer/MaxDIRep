
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import MaxPool2D,  BatchNormalization, Activation, Conv2DTranspose,Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.schedules import LearningRateSchedule






#Sampling layer for VAE (8,8,2)
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,7,7,2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class MyDecay(LearningRateSchedule):

    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta
    

 
    


class VAEGAN(object):
    def __init__(self, x_source_train, y_source_train, 
                 x_target_train, y_target_train, 
                 x_source_test, y_source_test, 
                 x_target_test, y_target_test, pretrained_model, resnet_decoder,
                 epochs=90):

        #source train and test dataset
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
        
        #Latent dim for AE/VAE
        self.latent_dim = (7,7,2048) #25
        #self.latent_dim_decoder = 1571
        
        
        self.pretrained_model = pretrained_model 
        self.resnet_decoder = resnet_decoder
        #self.encoder = encoder
        self.epochs = epochs 


        self.generator = Sequential([
            self.pretrained_model
            #GlobalAveragePooling2D(),
            #Dense(self.latent_dim, activation = "relu")
        ])

        self.classifier = Sequential([
            #Conv2D(filters=2048, kernel_size=(1,1)),
            GlobalAveragePooling2D(),
            Dense(256, activation = "relu"),
            Dense(self.n_classes, activation = "softmax")
        ])
        
        self.discriminator = Sequential([
           # Conv2D(filters=2048, kernel_size=(1,1)),
            GlobalAveragePooling2D(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(2, activation='softmax')
        ])
        
        
        
        
        encoder_inputs = keras.Input(shape=self.input_shape)
        # x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        # x = Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
        # x = Conv2D(8, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2D(32, 3, activation="relu", strides=4, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=4, padding="same")(x)
        #x = Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
        #x = Conv2D(8, 3, activation="relu", strides=2, padding="same")(x)
        
        z_mean = Conv2D(filters=2, kernel_size=(3,3),strides=2,padding='same',use_bias=False)(x)
        z_log_var = Conv2D(filters=2, kernel_size=(3,3),strides=2,padding='same',use_bias=False)(x)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
  
        
               
        self.decoder = Sequential([
            self.resnet_decoder
        ])
        
        self.predict_label = Sequential([
            self.generator,
            self.classifier
        ])
        
        self.classify_domain = Sequential([
            self.generator,
            self.discriminator
        ])
        

      
    
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        self.mse = tf.keras.losses.MeanSquaredError()
        

    
        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002

        
        self.task_optimizer= keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr, alpha=self.alpha),momentum=self.momentum, nesterov=True)
        self.gen_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/10., alpha=self.alpha), momentum=self.momentum, nesterov=True)
        self.disc_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/10., alpha=self.alpha))
        self.enc_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/5, alpha=self.alpha), momentum=self.momentum, nesterov=True)
        self.dec_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/5, alpha=self.alpha), momentum=self.momentum, nesterov=True)
        
   
        
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        self.train_recon_loss = tf.keras.metrics.Mean()
        self.train_kl_loss = tf.keras.metrics.Mean()
        self.train_target_recon_loss = tf.keras.metrics.Mean()
        self.train_target_kl_loss = tf.keras.metrics.Mean()

        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_domain_accuracy = tf.keras.metrics.CategoricalAccuracy()


        self.test_task_loss = tf.keras.metrics.Mean()
        self.test_disc_loss = tf.keras.metrics.Mean()
        self.test_gen_loss = tf.keras.metrics.Mean()
        self.test_recon_loss = tf.keras.metrics.Mean()
        self.test_kl_loss = tf.keras.metrics.Mean()
        self.test_target_task_loss = tf.keras.metrics.Mean()
        self.test_target_recon_loss = tf.keras.metrics.Mean()
        self.test_target_kl_loss = tf.keras.metrics.Mean()
        

        self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        

        
        self.batch_size = 16

        
    def KL(self, mean, log_var):
        loss_value = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        return tf.reduce_mean(tf.reduce_sum(loss_value, axis=1))
    

    def train_batch(self, x_source_train, y_source_train, x_target_train, epoch):
        
        source = np.tile([1,0], (x_source_train.shape[0], 1))
        target = np.tile([0,1], (x_target_train.shape[0], 1))
        domain_labels = np.concatenate([source, target], axis = 0)
        


        x_both = tf.concat([x_source_train, x_target_train], axis = 0)

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            
            #Forward pass
            y_class_pred = self.predict_label(x_source_train, training=True)
            y_domain_pred = self.classify_domain(x_both, training=True)
            
            
            DIrep_source = self.generator(x_source_train)  
            DIrep_target = self.generator(x_target_train)
    
            # print("DIrep source is {}".format(DIrep_source.shape))
            # print("DIrep target is {}".format(DIrep_target.shape))
            
            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_train, training = True)
            # print("DDrep source is {}".format(DDrep_source.shape))
            # print("d source is {}".format(domain_bit_source.shape))
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)
            
            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_train, training = True)
            # print("DDrep target is {}".format(DDrep_target.shape))
            # print("d target is {}".format(domain_bit_target.shape))
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)
            
            x_recon_source = self.decoder(concat_samples_source, training=True)
            x_recon_target = self.decoder(concat_samples_target, training=True)


            
            task_loss = self.loss(y_source_train, y_class_pred) 
            disc_loss = self.loss(domain_labels, y_domain_pred)  
            recon_loss_source = self.mse(x_source_train, x_recon_source)
            recon_loss_target = self.mse(x_target_train, x_recon_target)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)
            
            recon_loss = recon_loss_source + recon_loss_target 
            kl_loss =  kl_loss_source +  kl_loss_target
            
            
            
            total_loss = recon_loss + kl_loss*1/2000

            
            gen_loss = task_loss - disc_loss * 0.1  + recon_loss * 0.05
            
    
            
            
           
            
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        enc_grad = enc_tape.gradient(total_loss, self.encoder.trainable_variables) 
        dec_grad = dec_tape.gradient(total_loss, self.decoder.trainable_variables) 
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        self.enc_optimizer.apply_gradients(zip(enc_grad, self.encoder.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(dec_grad, self.decoder.trainable_variables))
        
            

        self.train_task_loss(task_loss)
        self.train_task_accuracy(y_source_train, y_class_pred)
        self.train_disc_loss(disc_loss)
        self.train_gen_loss(gen_loss)
        self.train_recon_loss(recon_loss_source)
        self.train_kl_loss(kl_loss_source)
        self.train_target_recon_loss(recon_loss_target)
        self.train_target_kl_loss(kl_loss_target)

            
            


        return
    
    
    def test_batch(self, x_source_test, y_source_test, x_target_test, y_target_test):
        

        
        
        
        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_source_test, training=False)
            y_target_class_pred = self.predict_label(x_target_test, training=False)
            
            DIrep_source = self.generator(x_source_test)  
            DIrep_target = self.generator(x_target_test)
            
            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_test, training = False)
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)
            
            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_test, training = False)
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)
             
            
            x_recon_source = self.decoder(concat_samples_source, training=False)
            x_recon_target = self.decoder(concat_samples_target, training=False)
    
    
            task_loss = self.loss(y_source_test, y_class_pred)
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
            recon_loss_source = self.mse(x_source_test, x_recon_source)
            recon_loss_target = self.mse(x_target_test, x_recon_target)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)
            
            

        self.test_task_loss(task_loss)
        self.test_recon_loss(recon_loss_source)
        self.test_kl_loss(kl_loss_source)
        self.test_task_accuracy(y_source_test, y_class_pred)
   

        self.test_target_task_loss(target_task_loss)
        self.test_target_recon_loss(recon_loss_target)
        self.test_target_kl_loss(kl_loss_target)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)

        return
    
    def log(self):
        
        
        log_format = 'C_loss train: {:.4f}, Acc train : {:.2f}\n'+ \
            'D_loss train: {:.4f}, recon_loss_source:{:.4f}, kl_loss_source:{:.4f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n'+ \
            'C_loss test source: {:.4f}, Acc test source: {:.2f}, recon_loss_source {:.4f}, kl_loss_source {:.4f}\n'+ \
            'C_loss test target: {:.4f}, Acc test target: {:.2f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.train_recon_loss.result(),
                 self.train_kl_loss.result(),
                 self.train_target_recon_loss.result(),
                 self.train_target_kl_loss.result(),
                 self.test_task_loss.result(),
                 self.test_task_accuracy.result()*100,
                 self.test_recon_loss.result(),
                 self.test_kl_loss.result(),
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100,
                 self.test_target_recon_loss.result(),
                 self.test_target_kl_loss.result())
        

        self.reset_metrics('train')
        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_recon_loss.reset_states()
            self.train_kl_loss.reset_states()
            self.train_target_recon_loss.reset_states()
            self.train_target_kl_loss.reset_states()
        
        if target == 'test':
            self.test_task_loss.reset_states()
            self.test_task_accuracy.reset_states()
            self.test_recon_loss.reset_states()
            self.test_kl_loss.reset_states()
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
            self.test_target_recon_loss.reset_states()
            self.test_target_kl_loss.reset_states()


        return
    
    
    def train(self):
        
        source_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).shuffle(len(self.y_source_train)).batch(self.batch_size)
        target_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_train, self.y_target_train)).shuffle(len(self.y_target_train)).batch(self.batch_size)
        
        source_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_test, self.y_source_test)).batch(self.batch_size)
        target_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_test, self.y_target_test)).batch(self.batch_size)
        

        
        
        for epoch in range(self.epochs):
            
            batches = 0 
            
            for (source_images, class_labels), (target_images, _) in zip(source_train_dataset, target_train_dataset):
                self.train_batch(source_images, class_labels, target_images, epoch)
            
 

                

            for (test_images, test_labels), (target_test_images, target_test_labels) in zip(source_test_dataset, target_test_dataset):
                self.test_batch(test_images, test_labels, target_test_images, target_test_labels)
                

            print('Epoch: {}'.format(epoch + 1))
            print(self.log())
            
        return self.generator, self.encoder, self.decoder

    
    
    
class MyDecay(LearningRateSchedule):

    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta
    

 
    


class VAEGANWithDiscriminatorLoss(object):
    def __init__(self, x_source_train, y_source_train, 
                 x_target_train, y_target_train, 
                 x_source_test, y_source_test, 
                 x_target_test, y_target_test, pretrained_model, resnet_decoder,
                 epochs=90):

        #source train and test dataset
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
        
        #Latent dim for AE/VAE
        self.latent_dim = (7,7,2048) #25
        #self.latent_dim_decoder = 1571
        
        
        self.pretrained_model = pretrained_model 
        self.resnet_decoder = resnet_decoder
        #self.encoder = encoder
        self.epochs = epochs 


        self.generator = Sequential([
            self.pretrained_model
            #GlobalAveragePooling2D(),
            #Dense(self.latent_dim, activation = "relu")
        ])

        self.classifier = Sequential([
            #Conv2D(filters=2048, kernel_size=(1,1)),
            GlobalAveragePooling2D(),
            Dense(256, activation = "relu"),
            Dense(self.n_classes, activation = "softmax")
        ])
        
        self.discriminator = Sequential([
           # Conv2D(filters=2048, kernel_size=(1,1)),
            GlobalAveragePooling2D(),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(1024),
            BatchNormalization(),
            Activation('relu'),
            Dense(2, activation='softmax')
        ])
        
        
        
        
        encoder_inputs = keras.Input(shape=self.input_shape)
        # x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        # x = Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
        # x = Conv2D(8, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2D(32, 3, activation="relu", strides=4, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=4, padding="same")(x)
        #x = Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
        #x = Conv2D(8, 3, activation="relu", strides=2, padding="same")(x)
        
        z_mean = Conv2D(filters=2, kernel_size=(3,3),strides=2,padding='same',use_bias=False)(x)
        z_log_var = Conv2D(filters=2, kernel_size=(3,3),strides=2,padding='same',use_bias=False)(x)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
  
        
               
        self.decoder = Sequential([
            self.resnet_decoder
        ])
        
        self.predict_label = Sequential([
            self.generator,
            self.classifier
        ])
        
        self.classify_domain = Sequential([
            self.generator,
            self.discriminator
        ])
        

      
    
        
      
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        self.mse = tf.keras.losses.MeanSquaredError()
        

    
        
        self.lr = 0.001 
        self.momentum = 0.9
        self.alpha = 0.0002

        
        self.task_optimizer= keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr, alpha=self.alpha),momentum=self.momentum, nesterov=True)
        self.gen_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/10., alpha=self.alpha), momentum=self.momentum, nesterov=True)
        self.disc_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/10., alpha=self.alpha))
        self.enc_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/5, alpha=self.alpha), momentum=self.momentum, nesterov=True)
        self.dec_optimizer = keras.optimizers.SGD(learning_rate=MyDecay(mu_0=self.lr/5, alpha=self.alpha), momentum=self.momentum, nesterov=True)
        
        
        # self.task_optimizer= keras.optimizers.Adam(0.0002,0.5) 
        # self.gen_optimizer = keras.optimizers.Adam(0.0002,0.5) 
        # self.disc_optimizer = keras.optimizers.Adam(0.0002,0.5) 
        # self.enc_optimizer = keras.optimizers.Adam(0.0002,0.5) 
        # self.dec_optimizer = keras.optimizers.Adam(0.0002,0.5) 
        
        
        self.train_task_loss = tf.keras.metrics.Mean()
        self.train_disc_loss = tf.keras.metrics.Mean()
        self.train_gen_loss = tf.keras.metrics.Mean()
        self.train_recon_loss = tf.keras.metrics.Mean()
        self.train_kl_loss = tf.keras.metrics.Mean()
        self.train_target_recon_loss = tf.keras.metrics.Mean()
        self.train_target_kl_loss = tf.keras.metrics.Mean()

        self.train_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_domain_accuracy = tf.keras.metrics.CategoricalAccuracy()


        self.test_task_loss = tf.keras.metrics.Mean()
        self.test_disc_loss = tf.keras.metrics.Mean()
        self.test_gen_loss = tf.keras.metrics.Mean()
        self.test_recon_loss = tf.keras.metrics.Mean()
        self.test_kl_loss = tf.keras.metrics.Mean()
        self.test_target_task_loss = tf.keras.metrics.Mean()
        self.test_target_recon_loss = tf.keras.metrics.Mean()
        self.test_target_kl_loss = tf.keras.metrics.Mean()
        

        self.test_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.test_target_task_accuracy = tf.keras.metrics.CategoricalAccuracy()
        

        
        self.batch_size = 16

        
    def KL(self, mean, log_var):
        loss_value = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        return tf.reduce_mean(tf.reduce_sum(loss_value, axis=1))
    

    def train_batch(self, x_source_train, y_source_train, x_target_train, epoch):
        
        source = np.tile([1,0], (x_source_train.shape[0], 1))
        target = np.tile([0,1], (x_target_train.shape[0], 1))
        domain_labels = np.concatenate([source, target], axis = 0)
        
        # invert_source = np.tile([0,1], (x_source_train.shape[0], 1))
        # invert_target = np.tile([1,0], (x_target_train.shape[0], 1))
        
        
        # domain_bit_shape_source=[x_source_train.shape[0], self.latent_dim[0], self.latent_dim[1], 1]
        # domain_bit_shape_target=[x_target_train.shape[0], self.latent_dim[0], self.latent_dim[1], 1]
        # domain_bit_source = tf.zeros(domain_bit_shape_source)
        # domain_bit_target = tf.ones(domain_bit_shape_target)
        
    
        
        #domain_bit_source = np.array(([0] * x_source_train.shape[0])).reshape((-1, 1))
        #domain_bit_target = np.array(([1] * x_target_train.shape[0])).reshape((-1, 1))
       # domain_labels = np.concatenate([source, target], axis = 0)
        #invert_domain_labels = np.concatenate([invert_source, invert_target], axis = 0)

        x_both = tf.concat([x_source_train, x_target_train], axis = 0)

        with tf.GradientTape() as task_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            
            #Forward pass
            y_class_pred = self.predict_label(x_source_train, training=True)
            y_domain_pred = self.classify_domain(x_both, training=True)
            
            
            DIrep_source = self.generator(x_source_train)  
            DIrep_target = self.generator(x_target_train)
    
            # print("DIrep source is {}".format(DIrep_source.shape))
            # print("DIrep target is {}".format(DIrep_target.shape))
            
            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_train, training = True)
            # print("DDrep source is {}".format(DDrep_source.shape))
            # print("d source is {}".format(domain_bit_source.shape))
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)
            
            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_train, training = True)
            # print("DDrep target is {}".format(DDrep_target.shape))
            # print("d target is {}".format(domain_bit_target.shape))
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)
            
            x_recon_source = self.decoder(concat_samples_source, training=True)
            x_recon_target = self.decoder(concat_samples_target, training=True)


            
            task_loss = self.loss(y_source_train, y_class_pred) 
            disc_loss = self.loss(domain_labels, y_domain_pred)  
            recon_loss_source = self.mse(x_source_train, x_recon_source)
            recon_loss_target = self.mse(x_target_train, x_recon_target)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)
            
            recon_loss = recon_loss_source + recon_loss_target 
            kl_loss =  kl_loss_source +  kl_loss_target
            
            
            
            total_loss = recon_loss + kl_loss*1/2000
            
            if epoch <= 25:
                disc_loss_coefficient = 0
            else: 
                disc_loss_coefficient = 0.1
            
            
            #disc_loss_coefficient = 2.0/(1+ math.exp(-1*epoch)) - 1.9
            
            #disc_loss_coefficient = 2.0/(1+ math.exp(-10*epoch*0.02)) - 1
            
            
            
            #gen_loss = task_loss - disc_loss * 0.1  + recon_loss * 0.05
            
            gen_loss = task_loss - disc_loss * disc_loss_coefficient  + recon_loss * 0.05
            
            
            
           
            
        
         # Compute gradients   
        task_grad = task_tape.gradient(task_loss, self.classifier.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        enc_grad = enc_tape.gradient(total_loss, self.encoder.trainable_variables) 
        dec_grad = dec_tape.gradient(total_loss, self.decoder.trainable_variables) 
        
        # Update weights 
        self.task_optimizer.apply_gradients(zip(task_grad, self.classifier.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables)) 
        self.disc_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
        self.enc_optimizer.apply_gradients(zip(enc_grad, self.encoder.trainable_variables))
        self.dec_optimizer.apply_gradients(zip(dec_grad, self.decoder.trainable_variables))
        
            

        self.train_task_loss(task_loss)
        self.train_task_accuracy(y_source_train, y_class_pred)
        self.train_disc_loss(disc_loss)
        self.train_gen_loss(gen_loss)
        self.train_recon_loss(recon_loss_source)
        self.train_kl_loss(kl_loss_source)
        self.train_target_recon_loss(recon_loss_target)
        self.train_target_kl_loss(kl_loss_target)

            
            


        return
    
    
    def test_batch(self, x_source_test, y_source_test, x_target_test, y_target_test):
        
        # domain_bit_shape_source=[x_source_test.shape[0], self.latent_dim[0],self.latent_dim[1], 1]
        # domain_bit_shape_target=[x_target_test.shape[0], self.latent_dim[0],self.latent_dim[1], 1]
        # domain_bit_source = tf.zeros(domain_bit_shape_source)
        # domain_bit_target = tf.ones(domain_bit_shape_target)
       
        #domain_bit_source = np.array(([0] * x_source_test.shape[0])).reshape((-1, 1))
        #domain_bit_target = np.array(([1] * x_target_test.shape[0])).reshape((-1, 1))
        
        
        
        with tf.GradientTape() as tape:
            y_class_pred = self.predict_label(x_source_test, training=False)
            y_target_class_pred = self.predict_label(x_target_test, training=False)
            
            DIrep_source = self.generator(x_source_test)  
            DIrep_target = self.generator(x_target_test)
            
            DDrep_mean_source, DDrep_log_var_source, DDrep_source = self.encoder(x_source_test, training = False)
            concat_samples_source = tf.concat([DIrep_source, DDrep_source], axis=3)
            
            DDrep_mean_target, DDrep_log_var_target, DDrep_target = self.encoder(x_target_test, training = False)
            concat_samples_target = tf.concat([DIrep_target, DDrep_target], axis=3)
             
            
            x_recon_source = self.decoder(concat_samples_source, training=False)
            x_recon_target = self.decoder(concat_samples_target, training=False)
    
    
            task_loss = self.loss(y_source_test, y_class_pred)
            target_task_loss = self.loss(y_target_test, y_target_class_pred)
            recon_loss_source = self.mse(x_source_test, x_recon_source)
            recon_loss_target = self.mse(x_target_test, x_recon_target)
            kl_loss_source = self.KL(DDrep_mean_source, DDrep_log_var_source)
            kl_loss_target = self.KL(DDrep_mean_target, DDrep_log_var_target)
            
            

        self.test_task_loss(task_loss)
        self.test_recon_loss(recon_loss_source)
        self.test_kl_loss(kl_loss_source)
        self.test_task_accuracy(y_source_test, y_class_pred)
   

        self.test_target_task_loss(target_task_loss)
        self.test_target_recon_loss(recon_loss_target)
        self.test_target_kl_loss(kl_loss_target)
        self.test_target_task_accuracy(y_target_test, y_target_class_pred)

        return
    
    def log(self):
        
        
        log_format = 'C_loss train: {:.4f}, Acc train : {:.2f}\n'+ \
            'D_loss train: {:.4f}, recon_loss_source:{:.4f}, kl_loss_source:{:.4f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n'+ \
            'C_loss test source: {:.4f}, Acc test source: {:.2f}, recon_loss_source {:.4f}, kl_loss_source {:.4f}\n'+ \
            'C_loss test target: {:.4f}, Acc test target: {:.2f}, recon_loss_target {:.4f}, kl_loss_target {:.4f}\n'

        message = log_format.format(
                 self.train_task_loss.result(),
                 self.train_task_accuracy.result()*100,
                 self.train_disc_loss.result(),
                 self.train_recon_loss.result(),
                 self.train_kl_loss.result(),
                 self.train_target_recon_loss.result(),
                 self.train_target_kl_loss.result(),
                 self.test_task_loss.result(),
                 self.test_task_accuracy.result()*100,
                 self.test_recon_loss.result(),
                 self.test_kl_loss.result(),
                 self.test_target_task_loss.result(),
                 self.test_target_task_accuracy.result()*100,
                 self.test_target_recon_loss.result(),
                 self.test_target_kl_loss.result())
        

        self.reset_metrics('train')
        self.reset_metrics('test')


        return message 
    
    def reset_metrics(self, target):

        if target == 'train':
            self.train_task_loss.reset_states()
            self.train_task_accuracy.reset_states()
            self.train_disc_loss.reset_states()
            self.train_recon_loss.reset_states()
            self.train_kl_loss.reset_states()
            self.train_target_recon_loss.reset_states()
            self.train_target_kl_loss.reset_states()
        
        if target == 'test':
            self.test_task_loss.reset_states()
            self.test_task_accuracy.reset_states()
            self.test_recon_loss.reset_states()
            self.test_kl_loss.reset_states()
            self.test_target_task_loss.reset_states()
            self.test_target_task_accuracy.reset_states()
            self.test_target_recon_loss.reset_states()
            self.test_target_kl_loss.reset_states()


        return
    
    
    def train(self):
        
        source_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_train, self.y_source_train)).shuffle(len(self.y_source_train)).batch(self.batch_size)
        target_train_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_train, self.y_target_train)).shuffle(len(self.y_target_train)).batch(self.batch_size)
        
        source_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_source_test, self.y_source_test)).batch(self.batch_size)
        target_test_dataset = tf.data.Dataset.from_tensor_slices((self.x_target_test, self.y_target_test)).batch(self.batch_size)
        
#         datagen_source = tf.keras.preprocessing.image.ImageDataGenerator(
#                 #featurewise_center=True,
#                 #featurewise_std_normalization=True,
#                 #width_shift_range=0.2,
#                 #eight_shift_range=0.2,
#                 horizontal_flip=True)
        
#         datagen_target = tf.keras.preprocessing.image.ImageDataGenerator(
#                 #featurewise_center=True,
#                 #featurewise_std_normalization=True,
#                 #rotation_range=20,
#                 #width_shift_range=0.2,
#                 #height_shift_range=0.2,
#                 horizontal_flip=True)
        
#         datagen_source.fit(self.x_source_train)
#         datagen_target.fit(self.x_target_train)
        
        # print(self.generator.summary())
        # print(self.classifier.summary())
        # print(self.discriminator.summary())
        # print(self.encoder.summary())
        # print(self.decoder.summary())
        
        
        for epoch in range(self.epochs):
            
            batches = 0 
            
            for (source_images, class_labels), (target_images, _) in zip(source_train_dataset, target_train_dataset):
                self.train_batch(source_images, class_labels, target_images, epoch)
            
            # for (source_images, class_labels), (target_images, _) in zip(datagen_source.flow(self.x_source_train, self.y_source_train, batch_size =self.batch_size), datagen_target.flow(self.x_target_train, self.y_target_train, batch_size =self.batch_size)):
            #     self.train_batch(source_images, class_labels, target_images, epoch)
            #     batches += 1
            #     if batches > self.x_source_train.shape[0] / self.batch_size:
            #         # we need to break the loop by hand because
            #         # the generator loops indefinitely
            #         break
        
                    
            #print(batches)

                

            for (test_images, test_labels), (target_test_images, target_test_labels) in zip(source_test_dataset, target_test_dataset):
                self.test_batch(test_images, test_labels, target_test_images, target_test_labels)
                

            print('Epoch: {}'.format(epoch + 1))
            print(self.log())
            
        return self.generator, self.encoder, self.decoder

    
    
    
    
    
    