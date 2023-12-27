from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from focal_loss import BinaryFocalLoss
import os
import tfimm
from vit_keras import vit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        print("Using GPU:", device.name)
    selected_gpu = 0
    tf.config.experimental.set_visible_devices(physical_devices[selected_gpu], 'GPU')
    print(f"Selected Using GPU: {physical_devices[selected_gpu].name}")
else:
    print("No GPU devices found.")
IMG_SIZE = 256
BATCH_SIZE = 16
base_dir = 'D:/Jiwoon/dataset/waper/total/'
classes = ['bad', 'good']

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    validation_split=0.25
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

def build_feature_extractor():
    '''['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtSmall', 'ConvNeXtTiny', 'ConvNeXtXLarge',
    'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
    'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0',
     'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S',
     'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge',
     'NASNetMobile', 'RegNetX002', 'RegNetX004', 'RegNetX006', 'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040',
     'RegNetX064', 'RegNetX080', 'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002', 'RegNetY004', 'RegNetY006',
     'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040', 'RegNetY064', 'RegNetY080', 'RegNetY120', 'RegNetY160',
     'RegNetY320', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101',
     'ResNetRS152', 'ResNetRS200', 'ResNetRS270', 'ResNetRS350', 'ResNetRS420', 'ResNetRS50', 'VGG16', 'VGG19',
     'Xception', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__',
     '__path__', '__spec__', '_sys', 'convnext', 'densenet', 'efficientnet', 'efficientnet_v2', 'imagenet_utils',
     'inception_resnet_v2', 'inception_v3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3', 'nasnet', 'regnet',
      'resnet', 'resnet50', 'resnet_rs', 'resnet_v2', 'vgg16', 'vgg19', 'xception']'''
    # feature_extractor = tfimm.create_model("convnext_base", pretrained="timm", nb_classes=0)
    # feature_extractor = keras.applications.DenseNet121(
    #     weights="imagenet",
    #     include_top=False,
    #     pooling="avg",
    #     input_shape=(IMG_SIZE, IMG_SIZE, 3),
    # )
    ''''BASE_URL', 'CONFIG_B', 'CONFIG_L', 'ConfigDict', 'ImageSizeArg',\
        'SIZES', 'WEIGHTS', '__annotations__', '__builtins__', '__cached__', \
        '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__',\
        'build_model', 'interpret_image_size', 'layers', 'load_pretrained', 'preprocess_inputs', \
        'tf', 'tx', 'typing', 'utils', 'validate_pretrained_top', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'warnings'''''
    print(dir(vit))
    cnn_feature_extractor = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # feature_extractor = tfimm.create_model("convnext_tiny", pretrained="timm", nb_classes=0)

    vit_feature_extractor = vit.vit_b16(
        image_size=IMG_SIZE,
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=1)
    cut_vit_model = tf.keras.Model(inputs=vit_feature_extractor.get_layer('reshape').output, outputs=vit_feature_extractor.output)

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    # preprocessed = preprocess_input(inputs)
    x = feature_extractor(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # print(99999999999999999999999999,outputs.shape)
    return keras.Model(inputs, outputs, name="feature_extractor")

def plot_metrics(history, metric, filepath):
    # Plot training & validation metric values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(filepath, f"{metric}.png"))
    plt.close()

def run_experiment():
    filepath = "./tmp/video_classifier"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath+'/transformer.h5', monitor="val_accuracy", save_weights_only=True, save_best_only=True, verbose=1, mode='max'
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1, mode='max'
    )
    # Reduce learning rate on plateau callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.1, patience=15, min_lr=1e-6, verbose=1, mode='max'
    )

    model = build_feature_extractor()

    model.compile(
        optimizer='adam',
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=BinaryFocalLoss(gamma=2),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    factor = 2
    history = model.fit(
        train_generator,
        # steps_per_epoch=factor * (train_generator.samples // train_generator.batch_size),
        # train_labels,
        # validation_split=0.15,
        validation_data=validation_generator,
        # validation_steps=factor * (validation_generator.samples // validation_generator.batch_size),
        epochs=200,
        callbacks=[checkpoint, early_stopping, reduce_lr],
    )
    plot_metrics(history, 'accuracy', filepath)
    plot_metrics(history, 'loss', filepath)
    plot_metrics(history, 'auc', filepath)

    _, accuracy, auc = model.evaluate(validation_generator)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test AUC: {round(auc, 4)}")
run_experiment()
# feature_extractor = build_feature_extractor()



