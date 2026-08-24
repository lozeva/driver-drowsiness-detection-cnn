"""
models.py

Model architectures for the Driver Drowsiness Detection binary
classification task: a custom CNN trained from scratch, and a
MobileNetV2-based transfer learning model.
"""

from tensorflow.keras import layers, models


# 1. Custom CNN (trained from scratch)
def build_custom_cnn(input_shape: tuple[int, int, int] = (160, 160, 1)) -> models.Sequential:
    """
    Build a custom convolutional neural network for binary drowsiness
    classification, trained from scratch (no pretrained weights).

    Architecture: three convolutional blocks (Conv2D -> BatchNormalization ->
    ReLU -> MaxPooling), increasing filter count with depth (32 -> 64 -> 128),
    followed by GlobalAveragePooling2D (instead of Flatten) to sharply reduce
    the parameter count before the dense classification head, a standard
    choice in modern CNN design (e.g. ResNet, MobileNet) that reduces
    overfitting risk compared to flattening directly into a large dense
    layer. Batch normalization is applied after each convolution (before
    the activation) to stabilize and speed up training by normalizing
    layer activations within each mini-batch. Dropout is applied for
    additional regularization, and the output uses a sigmoid activation
    for binary classification.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of a single input image (height, width, channels).
        Defaults to (160, 160, 1) - grayscale images matching the
        preprocessing pipeline in data_utils.py.

    Returns
    -------
    keras.Model
        Compiled-ready (but not yet compiled) Keras Sequential model.
    """
    custom_cnn_model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(128, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="custom_cnn")

    return custom_cnn_model


# 2. Transfer learning model (MobileNetV2)
def build_mobilenetv2_model(
    input_shape: tuple[int, int, int] = (160, 160, 1),
    freeze_base: bool = True,
) -> models.Model:
    """
    Build a transfer learning model using MobileNetV2 (pretrained on
    ImageNet) as a frozen feature extractor, with a new classification
    head trained on top for binary drowsiness classification.

    MobileNetV2 expects a 3-channel (RGB) input, while our preprocessing
    pipeline produces grayscale (1-channel) images. A Conv2D(3, 1x1) layer
    is used to learn a mapping from 1 channel to 3 channels, which is a
    more flexible alternative to naively repeating the single channel
    three times, and lets the model learn the most useful way to expand
    the grayscale signal into MobileNetV2's expected input format.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of a single input image (height, width, channels).
        Defaults to (160, 160, 1) - grayscale images matching the
        preprocessing pipeline in data_utils.py.
    freeze_base : bool
        If True (default), the MobileNetV2 base is frozen (its
        pretrained weights are not updated), so only the new
        classification head is trained. Set to False for a later
        fine-tuning stage, where some of the base layers are unfrozen
        and trained with a lower learning rate.

    Returns
    -------
    keras.Model
        Compiled-ready (but not yet compiled) Keras functional model.
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import Input, Model

    mobilenetv2_base = MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    mobilenetv2_base.trainable = not freeze_base

    model_input = Input(shape=input_shape)
    rgb_like_input = layers.Conv2D(3, kernel_size=1, padding="same", name="grayscale_to_rgb")(model_input)
    base_output = mobilenetv2_base(rgb_like_input)
    pooled_output = layers.GlobalAveragePooling2D()(base_output)
    dense_output = layers.Dense(128, activation="relu")(pooled_output)
    dropout_output = layers.Dropout(0.5)(dense_output)
    final_output = layers.Dense(1, activation="sigmoid")(dropout_output)

    mobilenetv2_model = Model(inputs=model_input, outputs=final_output, name="mobilenetv2_transfer")

    return mobilenetv2_model