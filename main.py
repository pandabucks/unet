import os
import cv2
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet import UNet

IMAGE_SIZE = 256


def normalize_x(image):
    image = image/127.5 - 1
    return image

def normalize_y(image):
    image = image/255
    return image

def denormalize_y(image):
    image = image * 255
    return image

def load_X(folder_path):
    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images[i] = normalize_x(image)
    return images, image_files

def load_Y(folder_path):
    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def train_unet():
    X_train, file_names = load_X(os.path.join(os.getcwd(), "images", "train", "rgb"))
    Y_train = load_Y(os.path.join(os.getcwd(), "images", "test", "label"))

    input_channel = 3
    output_channel = 1
    first_layer_filter_count = 64

    network = UNet(input_channel, output_channel, first_layer_filter_count)
    model = network.get_model()
    model.compile(loss=dice_coef_loss, optimizer=Adam(), metrics=[dice_coef])

    BATCH_SIZE = 12
    NUM_EPOCH = 20
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1)
    model.save_weights('unet_weights.hdf5')

def predict():
    X_test, file_names = load_X(os.path.join(os.getcwd(), "images", "test", "rgb"))

    input_channel = 3
    output_channel = 1
    first_layer_filter_count = 64
    network = UNet(input_channel, output_channel, first_layer_filter_count)
    model = network.get_model()
    model.load_weights('unet_weights.hdf5')
    BATCH_SIZE = 12
    Y_pred = model.predict(X_test, BATCH_SIZE)

    for i, y in enumerate(Y_pred):
        img = cv2.imread('testData' + os.sep + 'left_images' + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite('prediction' + str(i) + '.png', denormalize_y(y))


if __name__ == '__main__':
    # training unet model
    train_unet()
    predict()
