import sys
sys.path.append("../")
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
np_config.enable_numpy_behavior()

from models.office_31 import VAEGANImagenette
from preprocessing.office_31_data import load_imagenette, get_Xy
from sklearn.preprocessing import OneHotEncoder

if __name__ =="__main__":
    # The generator is a ResNet50 pretrained on ImageNet data
    # We use a pretrained decoder trained on the imagenette dataset
    # pretrained decoder is not included in the current repo since its size exceeds the max file size limit.
    decoder = tf.keras.models.load_model('../pretrained_decoder')

    # load imagenette data
    # VAE is also trained on the imagenette data during domain adaptation
    # Since the source and target data is very small, we do not want the weights of the pretrained VAEGAN to change drastically
    path_train = "../data/imagenette2-320/train"
    path_test = "../data/imagenette2-320/val"

    x_image_train, x_image_test = load_imagenette(path_train, path_test)

    #load target
    xA, yA = get_Xy("amazon")
    print('xA is : {0}'.format(xA.shape))
    print('yA is: {0}'.format(yA.shape))

    xD, yD = get_Xy("dslr")
    print('xD is : {0}'.format(xD.shape))
    print('yD is: {0}'.format(yD.shape))



    one = OneHotEncoder(sparse=False)
    ys = one.fit_transform(yD.reshape(-1, 1))
    yt = one.fit_transform(yA.reshape(-1, 1))

    xs = xD
    xt = xA

    print('x_source: {0}'.format(xs.shape))
    print('y_source: {0}'.format(ys.shape))
    print('x_target: {0}'.format(xt.shape))
    print('y_target: {0}'.format(yt.shape))

    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(xs, ys, test_size=0.10, random_state=42)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(xt, yt, test_size=0.10, random_state=42)

    print('x_source_train: {0}'.format(x_source_train.shape))
    print('y_source_train: {0}'.format(y_source_train.shape))
    print('x_target_train: {0}'.format(x_target_train.shape))
    print('y_target_train: {0}'.format(y_target_train.shape))
    print('x_source_test: {0}'.format(x_source_test.shape))
    print('y_source_test: {0}'.format(y_source_test.shape))
    print('x_target_test: {0}'.format(x_target_test.shape))
    print('y_target_test: {0}'.format(y_target_test.shape))


    n_classes = 31

    ResNet50, preprocess_input = Classifiers.get('resnet50')

    # build model
    # The generator is a ResNet50 pretrained on ImageNet data
    base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)

    #decoder = ResNet_decoder(True,"resnet50",(7,7,2051))


    model =VAEGANImagenette(x_source_train, y_source_train, x_target_train, y_target_train,
                x_image_train, x_image_test, x_source_test, y_source_test, x_target_test, y_target_test, base_model, decoder, epochs=60)

    generator, encoder, decoder = model.train()
