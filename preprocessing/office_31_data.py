
from __future__ import print_function, division
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
tf.autograph.set_verbosity(0)



def load_imagenette(path_train, path_test):
    x_train = []
    x_test = []

    for root, dirs, files in os.walk(path_train):
        for filename in files:
            if filename.endswith(".JPEG"):
                f = os.path.join(root, filename)
                image = Image.open(f).convert('RGB')
                image = image.resize((224, 224), Image.ANTIALIAS)
                image = np.array(image, dtype=int)
                x_train.append(image)

    for root, dirs, files in os.walk(path_test):
        for filename in files:
            if filename.endswith(".JPEG"):
                f = os.path.join(root, filename)
                image = Image.open(f).convert('RGB')
                image = image.resize((224, 224), Image.ANTIALIAS)
                image = np.array(image, dtype=int)
                x_test.append(image)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def get_Xy(domain, path_to_folder="data/office-31/"):
    path = path_to_folder + domain + "/images/"
    X = []
    y = []

    for r, d, f in os.walk(path):
        for direct in d:
            if not ".ipynb_checkpoints" in direct:
                for r, d, f in os.walk(os.path.join(path, direct)):
                    for file in f:
                        path_to_image = os.path.join(r, file)
                        if not ".ipynb_checkpoints" in path_to_image:
                            image = Image.open(path_to_image)
                            image = image.resize((224, 224), Image.ANTIALIAS)
                            image = np.array(image, dtype=int)
                            X.append(image)
                            y.append(direct)

    X = np.asarray(X)
    y = np.asarray(y)

    X = X.astype('float32') / 255.

    return X, y