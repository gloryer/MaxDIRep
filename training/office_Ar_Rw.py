import tensorflow as tf
from sklearn.model_selection import train_test_split
from classification_models.tfkeras import Classifiers
from sklearn.preprocessing import OneHotEncoder

from models.office import VAEGANWithDiscriminatorLoss 
from preprocessing.office_31_data import get_Xy



if __name__ == "__main__":

    xA, yA = get_Xy("Art")
    xR, yR = get_Xy("Real World")

    decoder = tf.keras.models.load_model('../pretrained_decoder')

    one = OneHotEncoder(sparse=False)

    ys = one.fit_transform(yA.reshape(-1, 1))
    yt = one.fit_transform(yR.reshape(-1, 1))

    xs = xA
    xt = xR

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


    n_classes = 65

    ResNet50, preprocess_input = Classifiers.get('resnet50')

    # build model
    base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)

    #decoder = ResNet_decoder(True,"resnet50",(7,7,2051))


    model =VAEGANWithDiscriminatorLoss(x_source_train, y_source_train, x_target_train, y_target_train, 
                x_source_test, y_source_test, x_target_test, y_target_test, base_model, decoder, epochs=50)

    generator, encoder, decoder = model.train()
