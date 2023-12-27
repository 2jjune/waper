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
from sklearn.svm import SVC

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

def train_gmm(features, labels):
    # GMM 학습
    gmm = GaussianMixture(n_components=2)
    gmm.fit(features, labels)
    return gmm

def predict_gmm(gmm, features):
    # GMM을 사용하여 테스트 데이터에 대한 확률을 계산하고 분류합니다.
    probs = gmm.predict_proba(features)
    return np.argmax(probs, axis=1)
def train_svm(features, labels):
    svm = SVC(probability=True) # 확률을 반환하기 위해 probability=True로 설정
    svm.fit(features, labels)
    return svm

def predict_svm(svm, features):
    return svm.predict(features)
def build_feature_extractor():
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

    return keras.Model(inputs, x, name="waper_model")

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
def extract_features(generator, model, sample_count):
    features = np.zeros((sample_count, 1536)) # 예: Swin Transformer의 출력 크기가 1536이라고 가정
    labels = np.zeros(sample_count)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels

from sklearn.ensemble import RandomForestClassifier

def train_rf(features, labels):
    rf = RandomForestClassifier()
    rf.fit(features, labels)
    return rf

def predict_rf(rf, features):
    return rf.predict(features)

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def voting(features, labels):
    svm = SVC(probability=True, kernel='linear')
    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    voting_clf = VotingClassifier(estimators=[('svm', svm), ('rf', rf)], voting='soft')

    # 모델 학습
    voting_clf.fit(features, labels)
    return voting_clf

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

    feature_extractor_model = build_feature_extractor()
    feature_extractor_model.summary()

    train_features, train_labels = extract_features(train_generator, feature_extractor_model, train_generator.samples)
    validation_features, validation_labels = extract_features(validation_generator, feature_extractor_model,
                                                              validation_generator.samples)

    gmm_model = train_gmm(train_features, train_labels)

    predicted_labels = predict_gmm(gmm_model, validation_features)

    # 분류 결과를 평가할 수 있습니다.
    accuracy = np.mean(predicted_labels == validation_labels)
    print(f"gmm Accuracy: {accuracy:.2f}")

    svm_model = train_svm(train_features, train_labels)

    # 학습된 SVM을 사용하여 검증 데이터를 분류합니다.
    predicted_labels = predict_svm(svm_model, validation_features)

    # 분류 결과를 평가할 수 있습니다.
    accuracy = np.mean(predicted_labels == validation_labels)
    print(f"svm Accuracy: {accuracy:.2f}")

    rf_model = train_rf(train_features, train_labels)
    # 학습된 Random Forest를 사용하여 검증 데이터를 분류합니다.
    predicted_labels_rf = predict_rf(rf_model, validation_features)
    # Random Forest의 분류 결과를 평가합니다.
    accuracy_rf = np.mean(predicted_labels_rf == validation_labels)
    print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

    voting_model = voting(train_features, train_labels)
    predicted_labels = voting_model.predict(validation_features)
    accuracy = accuracy_score(validation_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")

    #
    #
    # model.load_weights(filepath+'/gaussian_original_swin_large_384_transformer.h5')
    #
    # y_true = validation_generator.classes
    #
    # start_time = time.time()
    # y_pred = model.predict(validation_generator).ravel()
    # end_time = time.time()  # predict 후 현재 시간 측정
    # elapsed_time = end_time - start_time  # 걸린 시간 계산
    #
    # print(f"Model prediction took {elapsed_time:.2f} seconds")
    # plot_roc_auc(y_true, y_pred, filepath)
    #
    # plot_metrics(history, 'acc', filepath)
    # plot_metrics(history, 'loss', filepath)
    # plot_metrics(history, 'auc', filepath)
    # plot_metrics(history, 'f1_score', filepath)
    # _, accuracy, auc, f1 = model.evaluate(validation_generator)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test AUC: {round(auc, 4)}")
    # print(f"Test f1_score: {np.round(f1, 4)}")
run_experiment()
# feature_extractor = build_feature_extractor()



