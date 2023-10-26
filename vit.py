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
from Swin_Transformer_TF.swintransformer import SwinTransformer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
from sklearn.mixture import GaussianMixture

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
IMG_SIZE = 384
BATCH_SIZE = 8
# base_dir = 'D:/Jiwoon/dataset/waper/origin/'
base_dir = 'D:/Jiwoon/dataset/waper/cropped_300/'
classes = ['bad', 'good']

datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=5,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    # fill_mode='reflect',
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
    shuffle=False,
    subset='validation'
)


def patch_embedding_with_mobilenet(input_tensor, patch_size=16):
    # 1. 이미지를 16x16 패치로 분할
    patches = tf.image.extract_patches(
        input_tensor,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    # 패치의 형태를 조절 (batch_size, num_patches, patch_size, patch_size, channels)
    patches = tf.reshape(patches, [-1, 24, 24, 16, 16, 3])
    patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
    patches_resized = tf.image.resize(patches, [32, 32], method='bilinear')
    # 2. 각 패치에 대한 MobileNetV2 처리
    mobile_model = keras.applications.MobileNetV2(input_shape=(patch_size*2, patch_size*2, 3), include_top=False, weights='imagenet')
    # MobileNetV2의 출력을 768차원의 벡터로 변환하기 위한 추가 레이어
    projection_layer = tf.keras.layers.Dense(units=768, activation='linear')

    patch_embeddings = []
    for i in range(576):  # 576개의 패치를 반복합니다.
        patch = tf.expand_dims(patches_resized[i], 0)
        embedding = mobile_model(patch)
        embedding = tf.keras.layers.GlobalAveragePooling2D()(embedding)
        embedding_tensor = projection_layer(embedding)
        patch_embeddings.append(embedding_tensor)

    patch_embeddings = tf.stack(patch_embeddings, axis=0)
    embedding_tensor = tf.reshape(patch_embeddings, [-1,576,768])
    # MobileNetV2의 출력을 768차원의 벡터로 변환하기 위한 추가 레이어


    return embedding_tensor

class GaussianLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.add_weight(name='mean',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        self.log_var = self.add_weight(name='log_var',
                                       shape=(input_shape[-1],),
                                       initializer='zeros',
                                       trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        log_likelihood = -0.5 * (self.log_var + tf.square(x - self.mean) / tf.exp(self.log_var))
        return log_likelihood
class PlanarFlow(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(PlanarFlow, self).__init__(**kwargs)
        self.dim = dim
        self.u = self.add_weight(shape=(1, self.dim), initializer="random_normal", trainable=True)
        self.w = self.add_weight(shape=(1, self.dim), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(), initializer="zeros", trainable=True)

    def call(self, z):
        uw = tf.matmul(self.u, self.w, transpose_b=True)
        m_uw = -1 + tf.nn.softplus(uw)
        u_hat = self.u + (m_uw - uw) * self.w / tf.norm(self.w, ord=2)
        zwb = tf.matmul(z, self.w, transpose_b=True) + self.b
        f_z = z + u_hat * tf.tanh(zwb)
        return f_z
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

    ''''BASE_URL', 'CONFIG_B', 'CONFIG_L', 'ConfigDict', 'ImageSizeArg',\
        'SIZES', 'WEIGHTS', '__annotations__', '__builtins__', '__cached__', \
        '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__',\
        'build_model', 'interpret_image_size', 'layers', 'load_pretrained', 'preprocess_inputs', \
        'tf', 'tx', 'typing', 'utils', 'validate_pretrained_top', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'warnings'''''
    # print(dir(vit))
    print(90909090909090090990909090909090909)
    # print(tfimm.list_models())
    '''
    swin_base_patch4_window7_224', 'swin_base_patch4_window7_224_in22k', 'swin_base_patch4_window12_384',
     'swin_base_patch4_window12_384_in22k', 'swin_large_patch4_window7_224', 'swin_large_patch4_window7_224_in22k',
      'swin_large_patch4_window12_384', 'swin_large_patch4_window12_384_in22k', 'swin_small_patch4_window7_224',
       'swin_tiny_patch4_window7_224'
    '''
    '''
     'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 
     'vit_base_patch16_224_miil', 'vit_base_patch16_224_miil_in21k', 'vit_base_patch16_384', 'vit_base_patch16_sam_224', 
     'vit_base_patch32_224', 'vit_base_patch32_224_in21k', 'vit_base_patch32_384', 'vit_base_patch32_sam_224', 
     'vit_base_r50_s16_224_in21k', 'vit_base_r50_s16_384', 'vit_huge_patch14_224_in21k', 'vit_large_patch16_224',
      'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224', 'vit_large_patch32_224_in21k',
       'vit_large_patch32_384', 'vit_large_r50_s32_224', 'vit_large_r50_s32_224_in21k', 'vit_large_r50_s32_384', 
       'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_small_patch16_384', 'vit_small_patch32_224',
        'vit_small_patch32_224_in21k', 'vit_small_patch32_384', 'vit_small_r26_s32_224', 'vit_small_r26_s32_224_in21k', 
        'vit_small_r26_s32_384', 'vit_tiny_patch16_224', 'vit_tiny_patch16_224_in21k', 'vit_tiny_patch16_384', 
        'vit_tiny_r_s16_p8_224', 'vit_tiny_r_s16_p8_224_in21k', 'vit_tiny_r_s16_p8_384'
    '''

    '''
    'convnext_base', 'convnext_base_384_in22ft1k', 'convnext_base_in22ft1k', 'convnext_base_in22k', 
    'convnext_large', 'convnext_large_384_in22ft1k', 'convnext_large_in22ft1k', 'convnext_large_in22k',
     'convnext_small', 'convnext_small_384_in22ft1k', 'convnext_small_in22ft1k', 'convnext_small_in22k',
      'convnext_tiny', 'convnext_tiny_384_in22ft1k', 'convnext_tiny_in22ft1k', 'convnext_tiny_in22k', 
      'convnext_xlarge_384_in22ft1k', 'convnext_xlarge_in22ft1k', 'convnext_xlarge_in22k'
    '''
    #  vit_small_patch16_224      vit_small_patch32_224   vit_tiny_patch16_224
    #   vit_tiny_patch16_384    vit_large_patch32_384   vit_small_patch16_384   vit_small_patch32_384
    # convnext_tiny  convnext_base  convnext_large  convnext_small  convnext_base_in22k  convnext_xlarge_in22k  convnext_xlarge_in22ft1k
    # feature_extractor = tfimm.create_model("vit_large_patch16_224", pretrained="timm", nb_classes=0)
    # feature_extractor = tfimm.create_model("vit_small_patch16_224", pretrained="timm", nb_classes=0)

    # feature_extractor = tfimm.create_model("convnext_base", pretrained="timm",nb_classes=0)
    # ##'swin_tiny_224', swin_small_224  swin_base_224  swin_base_384  swin_large_224  swin_large_384
    feature_extractor = SwinTransformer('swin_large_384', include_top=False, pretrained=True)
    # feature_extractor = vit.vit_b16(
    #         image_size=IMG_SIZE,
    #         activation='sigmoid',
    #         pretrained=True,
    #         include_top=False,
    #         pretrained_top=False,
    #         classes=1)
    # feature_extractor.summary()


    # dummy_input = tf.random.normal(shape=(1,384,384,3))  # 예를 들어 (1, 224, 224, 3)과 같은 형상
    # # 레이어에 더미 데이터 전달
    # _ = feature_extractor(dummy_input)
    # first_layer = feature_extractor.layers[2]
    # print('*'*100)
    # print(type(first_layer))

    # kernel_shape = first_layer.weights[0].shape
    # input_channels = kernel_shape[-2]
    # output_channels = kernel_shape[-1]

    # first_layer_output = first_layer(dummy_input)
    # config = first_layer.get_config()
    # print("Kernel size:", config["kernel_size"])
    # print("Strides:", config["strides"])
    # print("Padding:", config["padding"])
    # print("Activation:", config["activation"])
    # 입력 및 출력 형상 출력
    # print("First Layer Input Shape:", dummy_input.shape)
    # print("First Layer Output Shape:", first_layer_output.shape)
    # #---------------------------------------------
    # # 입력 및 출력 형상 출력
    # print("First Layer Input Channels:", input_channels)
    # print("First Layer Output Channels:", output_channels)
    #
    # print("First Layer Name:", first_layer.name)
    # print("Input Shape:", first_layer.input_shape)
    # print("Output Shape:", first_layer.output_shape)
    # print("Config:", first_layer.get_config())
    # print('*'*100)

    # feature_extractor = vit.vit_b16(
    #     image_size=IMG_SIZE,
    #     activation='sigmoid',
    #     pretrained=True,
    #     include_top=False,
    #     pretrained_top=False,
    #     classes=1)
    # mobilenet = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False
    # --------------------mobilenet+vit 할라고 했던거
    # new_layers = []
    #
    # for layer in feature_extractor.layers:
    #     if layer.name == "embedding":  # 패치 임베딩 레이어의 이름
    #         print(layer.output_shape,layer.output_shape,layer.output_shape,layer.output_shape)
    #         # new_layer = patch_embedding_with_mobilenet(layer.output_shape)
    #         new_layer = lambda tensor: patch_embedding_with_mobilenet(tensor)
    #     else:
    #         new_layer = layer.__class__.from_config(layer.get_config())
    #     new_layers.append(new_layer)
    #
    # x = inputs
    # for layer in new_layers:
    #     x = layer(x)
    #----------------------------------------
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    # preprocessed = preprocess_input(inputs)
    x = feature_extractor(inputs)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = GaussianLayer()(x)
    x = PlanarFlow(dim=x.shape[-1], name="flowflow")(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # print(99999999999999999999999999,outputs.shape)
    return keras.Model(inputs, outputs, name="waper_model")

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


def plot_roc_auc(y_true, y_pred, filepath):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
    plt.figure(figsize=(10, 6))
    plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, 'b', label='(AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [1, 1], 'y--')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.grid(True)
    plt.savefig(os.path.join(filepath, "roc_auc_curve.png"))
    plt.close()

def run_experiment():
    filepath = "./tmp/video_classifier"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    # checkpoint = keras.callbacks.ModelCheckpoint(
    #     filepath+'/gaussian_original_swin_large_384_transformer.h5', monitor="val_acc", save_weights_only=True, save_best_only=True, verbose=1, mode='max'
    # )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=20, restore_best_weights=True, verbose=1, mode='max'
    )
    # Reduce learning rate on plateau callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc', factor=0.1, patience=15, min_lr=1e-6, verbose=0, mode='max'
    )

    model = build_feature_extractor()
    model.summary()
    for layer in model.layers:
        print(layer.name)
    learning_rate = 1e-4
    weight_decay = 0.0001
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer='adam',
        # optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=BinaryFocalLoss(gamma=2),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5),
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
        callbacks=[early_stopping, reduce_lr],
    )
    model.load_weights(filepath+'/gaussian_original_swin_large_384_transformer.h5')

    y_true = validation_generator.classes

    start_time = time.time()
    y_pred = model.predict(validation_generator).ravel()
    end_time = time.time()  # predict 후 현재 시간 측정
    elapsed_time = end_time - start_time  # 걸린 시간 계산

    print(f"Model prediction took {elapsed_time:.2f} seconds")
    plot_roc_auc(y_true, y_pred, filepath)

    plot_metrics(history, 'acc', filepath)
    plot_metrics(history, 'loss', filepath)
    plot_metrics(history, 'auc', filepath)
    plot_metrics(history, 'f1_score', filepath)
    _, accuracy, auc, f1 = model.evaluate(validation_generator)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test AUC: {round(auc, 4)}")
    print(f"Test f1_score: {np.round(f1, 4)}")
run_experiment()
# feature_extractor = build_feature_extractor()



