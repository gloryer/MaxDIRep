import tensorflow as tf
from sklearn.model_selection import train_test_split
from classification_models.tfkeras import Classifiers
from sklearn.preprocessing import OneHotEncoder
import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from models.office_home import MaxDIRep
from preprocessing.office_home_data import get_Xy



if __name__ == "__main__":

    xA, yA = get_Xy("Art")
    xC, yC = get_Xy("Clipart")

    decoder = tf.keras.models.load_model('pretrained_decoder')

    one = OneHotEncoder(sparse=False)

    ys = one.fit_transform(yA.reshape(-1, 1))
    yt = one.fit_transform(yC.reshape(-1, 1))

    xs = xA
    xt = xC

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

  
    base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)


    model =MaxDIRep(x_source_train, y_source_train, x_target_train, y_target_train, 
                x_source_test, y_source_test, x_target_test, y_target_test, base_model, decoder, epochs=50)

    generator, encoder, decoder = model.train()