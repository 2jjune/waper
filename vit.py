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
import argparse

### 경고 무시 및 GPU 출력
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

### 가우시안 레이어(사용안함)
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
### 여기까지 사용 안함

###
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

    if model_type == "vit_large":
        feature_extractor = tfimm.create_model("vit_large_patch16_224", pretrained="timm", nb_classes=0)
    elif model_type == "vit_small":
        feature_extractor = tfimm.create_model("vit_small_patch16_224", pretrained="timm", nb_classes=0)
    elif model_type == "convnext":
        feature_extractor = tfimm.create_model("convnext_base", pretrained="timm", nb_classes=0)
    elif model_type == "swin":
        feature_extractor = SwinTransformer('swin_large_384', include_top=False, pretrained=True)
    else:
        raise ValueError("Invalid model type selected.")

    ### TENSORFLOW VIT LOAD 함수(사용안함)
    # feature_extractor = vit.vit_b16(
    #         image_size=IMG_SIZE,
    #         activation='sigmoid',
    #         pretrained=True,
    #         include_top=False,
    #         pretrained_top=False,
    #         classes=1)
    # feature_extractor.summary()

    ### 각 레이어 TRAIN FALSE 설정
    for layer in feature_extractor.layers:
        layer.trainable = False

    #----------------모델 생성------------------------
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = feature_extractor(inputs)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = GaussianLayer()(x)
    # x = PlanarFlow(dim=x.shape[-1], name="flowflow")(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # print(99999999999999999999999999,outputs.shape)
    return keras.Model(inputs, outputs, name="waper_model")

### 각 평가지표(ACC, LOSS 등) 그래프 저장
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
    plt.savefig(os.path.join(filepath, f"origin_convnextbase_{metric}.png"))
    plt.close()

### ROCAUC 그래프 저장
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
    plt.savefig(os.path.join(filepath, "origin_convnextbase_roc_auc_curve.png"))
    plt.close()


### 학습 파라미터 설정 및 학습, 검증
def run_experiment():
    ### 이미지 AUGMENTATION
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=5,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        # fill_mode='reflect',
        validation_split=validation_rate
    )

    ###TRAIN, VALIDATION DATA 설정
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

    filepath = "./tmp/video_classifier"
    if not os.path.exists(filepath):#저장 경로 설정
        os.makedirs(filepath)

    ### CALLBACK 함수 설정
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath+'/tttswin_large_384_transformer.h5', monitor="val_acc", save_weights_only=True, save_best_only=True, verbose=1, mode='max'
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc', patience=20, restore_best_weights=True, verbose=1, mode='max'
    )
    # Reduce learning rate on plateau callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc', factor=0.1, patience=15, min_lr=1e-6, verbose=0, mode='max'
    )
    #-------------------

    model = build_feature_extractor()# 모델 불러오기
    model.summary()
    for layer in model.layers:
        print(layer.name)

    # 학습 파라미터 설정
    learning_rate = 1e-4
    weight_decay = 0.0001
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer='adam',
        # optimizer=optimizer,
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=BinaryFocalLoss(gamma=2),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        ],
    )

    ##학습
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=200,
        callbacks=[early_stopping, reduce_lr, checkpoint],
    )

    # 저장된 모델 불러오기
    model.load_weights(filepath+'/tttswin_large_384_transformer.h5')

    y_true = validation_generator.classes #정답값 뽑아오기

    start_time = time.time()
    y_pred = model.predict(validation_generator).ravel()
    end_time = time.time()  # predict 후 현재 시간 측정
    elapsed_time = end_time - start_time  # 걸린 시간 계산

    print(f"Batch : {BATCH_SIZE} Model prediction took {elapsed_time:.2f} seconds") # 모든 검증 데이터셋을 PREDICT하는데 걸린 시간
    plot_roc_auc(y_true, y_pred, filepath)# ROCAUC그래프 저장

    plot_metrics(history, 'acc', filepath)# ACC 그래프 저장
    plot_metrics(history, 'loss', filepath)# LOSS 그래프 저장
    plot_metrics(history, 'auc', filepath)# AUC 그래프 저장
    plot_metrics(history, 'f1_score', filepath)# F1 그래프 저장
    _, accuracy, auc, f1 = model.evaluate(validation_generator)# 모델 EVALUATE
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test AUC: {round(auc, 4)}")
    print(f"Test f1_score: {np.round(f1, 4)}")

if __name__=="__main__":
    # 명령줄 인수 파싱을 위한 파서 생성
    parser = argparse.ArgumentParser(description="Run the VIT model training and evaluation.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for the model.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--validate_rate", type=float, default=0.25, help="Batch size for training.")
    parser.add_argument("--data_dir", type=str, default="./", help="Directory of the dataset.")
    parser.add_argument("--model_type", type=str, default="vit_small", help="Type of the model to use.",
                        choices=['vit_large', 'vit_small', 'convnext', 'swin'])

    # 인수 파싱
    args = parser.parse_args()

    # 파싱된 인수를 변수에 할당
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch
    validation_rate = args.validate_rate
    base_dir = args.data_dir
    model_type = args.model_type

    run_experiment()

