

from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import MaxPool2D,  BatchNormalization, Activation, Conv2DTranspose,Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential




def ResNet_decoder(use_bias,
                   model_name="resnet",
                   input_shape=None
                   ):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Args:
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    Returns:
      A `keras.Model` instance.
    """
    img_input = Input(shape=input_shape)
    x = stack1(img_input, 512, 3, name="conv5")
    x = stack1(x, 256, 4, name="conv4")
    x = stack1(x, 128, 6, name="conv3")
    x = stack_last(x, 64, 3, stride1=1, name="conv2")

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(3, 7, strides=2, use_bias=use_bias, padding="same", activation="sigmoid", name="conv1_conv")(x)

    # Create model.
    model = Model(img_input, x, name=model_name)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3

    if conv_shortcut:
        shortcut = layers.Conv2DTranspose(
            filters, 1, strides=stride, name=name + "_0_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2DTranspose(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn"
    )(x)
    x = layers.Conv2DTranspose(
        filters, kernel_size, padding="SAME", name=name + "_2_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2DTranspose(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the last layer in the last block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, name=name + "_block1")
    for i in range(2, blocks):
        x = block1(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    x = block1(x, filters, stride=stride1, name=name + "_block" + str(blocks))
    return x


def stack_last(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks for the last stack
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the last layer in the last block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, name=name + "_block1")
    for i in range(2, blocks):
        x = block1(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    x = block1(x, filters, stride=stride1, conv_shortcut=False, name=name + "_block" + str(blocks))
    return x




