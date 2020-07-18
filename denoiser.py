import math
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import initializers
from tensorflow import keras


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def add_noise_and_clip_data(data):
   noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
   data = data + noise
   data = np.clip(data, 0., 1.)
   return data


def generate_noisy_images():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    noisy_train = add_noise_and_clip_data(trainX)
    noisy_test = add_noise_and_clip_data(testX)

    plt.figure(figsize=(18, 18))

    plt.subplot(541)
    plt.imshow(testX[100])
    plt.title('Original image')
    plt.subplot(542)
    plt.imshow(noisy_test[100])
    plt.title('Image with noise')

    plt.subplot(543)
    plt.imshow(testX[200])
    plt.title('Original image')
    plt.subplot(544)
    plt.imshow(noisy_test[200])
    plt.title('Image with noise')

    plt.subplot(545)
    plt.imshow(testX[302])
    plt.title('Original image')
    plt.subplot(546)
    plt.imshow(noisy_test[302])
    plt.title('Image with noise')

    plt.subplot(547)
    plt.imshow(testX[400])
    plt.title('Original image')
    plt.subplot(548)
    plt.imshow(noisy_test[400])
    plt.title('Image with noise')

    # plt.subplot(641)
    # plt.imshow(testX[40])
    # plt.title('Original image')
    # plt.subplot(642)
    # plt.imshow(noisy_test[40])
    # plt.title('Image with noise')

    # plt.savefig('1_b_1.png')

    plt.show()

def residual_block(x: Tensor, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= 1,
               filters=filters,
               padding="same",
                kernel_initializer=initializers.RandomNormal(stddev=0.001))(x)
    y = ReLU()(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same",
                kernel_initializer=initializers.RandomNormal(stddev=0.001))(y)
    out = Add()([x, y])
    return out
    # return y


def create_res_net():
    inputs = Input(shape=(32, 32, 3))
    t = Conv2D(kernel_size=9,
               strides=1,
               filters=64,
               padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.001))(inputs)
    t = Conv2D(kernel_size=5,
               strides=1,
               filters=32,
               padding="same",
               kernel_initializer=initializers.RandomNormal(stddev=0.001))(t)

    for i in range(5):
        t = residual_block(t, filters=32)

    # t = AveragePooling2D(4)(t)
    # t = Flatten()(t)
    # outputs = Dense(10, activation='softmax')(t)
    t = Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding='same',
               kernel_initializer=initializers.RandomNormal(stddev=0.001))(t)
    # t = Subtract()([inputs, t])   # input - noise

    model = Model(inputs, outputs=t)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse'
    )

    return model

def compute_psnr(img1, img2):
    height, width, channel = img1.shape
    size = height * width

    mseb = ((img1[:, :, 0] - img2[:, :, 0]) ** 2).sum()
    mseg = ((img1[:, :, 1] - img2[:, :, 1]) ** 2).sum()
    mser = ((img1[:, :, 2] - img2[:, :, 2]) ** 2).sum()

    MSE = (mseb + mseg + mser) / (3 * size)
    psnr = 20 * math.log10(1) - 10 * math.log10(MSE)
    return round(psnr, 2)

def psnr_test_set():
    trainX, trainY, testX, testY = load_dataset()
    # noisy_train, noisy_test, gauss_train = add_noise(trainX, testX)
    trainX, testX = prep_pixels(trainX, testX)
    noisy_train = add_noise_and_clip_data(trainX)
    noisy_test= add_noise_and_clip_data(testX)
    mo = load_model('without_res.h5')
    list_of_psnr = []
    for idx in range(0, len(noisy_test)):
      img_test = noisy_test[idx]
      img_test_new = np.reshape(img_test,(1,32,32,3))
      img = mo.predict(img_test_new)
      img = np.reshape(img,(32,32,3))
      img_predicted = np.clip(img, 0., 1.)
      list_of_psnr.append(compute_psnr(testX[idx], img_predicted))
    mean = sum(list_of_psnr) / len(list_of_psnr)
    variance = sum([((x - mean) ** 2) for x in list_of_psnr]) / len(list_of_psnr)
    res = variance ** 0.5
    print('The mean over test set is:', mean)
    print('The standard deviation over test set is', res)


def denoise_images():
    trainX, trainY, testX, testY = load_dataset()
    # noisy_train, noisy_test, gauss_train = add_noise(trainX, testX)
    trainX, testX = prep_pixels(trainX, testX)
    noisy_train = add_noise_and_clip_data(trainX)
    noisy_test = add_noise_and_clip_data(testX)
    mo = load_model('without_res.h5')

    # 100,1000,999,
    plt.figure(figsize=(18, 18))
    idx = 19
    img_test = noisy_test[idx]
    img_test_new = np.reshape(img_test, (1, 32, 32, 3))
    img = mo.predict(img_test_new)
    img = np.reshape(img, (32, 32, 3))
    img_predicted = np.clip(img, 0., 1.)
    # print(compute_psnr(trainX[idx],noisy_train[idx]))
    print(compute_psnr(testX[idx], img_predicted))
    # print(compute_psnr(noisy_train[idx], img_predicted))

    plt.subplot(330 + 1 + 0)
    plt.imshow(testX[idx])
    plt.title('Original image')
    plt.subplot(330 + 1 + 1)
    plt.imshow(noisy_test[idx])
    plt.title('Image with noise')
    plt.subplot(330 + 1 + 2)
    plt.imshow(img_predicted)
    plt.title('Denoised Image')
    # plt.savefig('sample_from_test.png')
    plt.show()

    # OLD CODE FOR PLOTTING, CAN BE USED UP LATER
    # pyplot.figure(figsize=(18, 18))

    # pyplot.subplot(330 + 1 + 0)
    # pyplot.imshow(rgb2gray(trainX[1]))

    # pyplot.subplot(330 + 1 + 1)
    # pyplot.imshow(rgb2gray(np.uint8(gauss_train[1])))

    # pyplot.subplot(330 + 1 + 2)
    # pyplot.imshow(rgb2gray(np.uint8(noisy_train[1])))

    # pyplot.show()

generate_noisy_images()
m = create_res_net()
print(m.summary())
psnr_test_set()
denoise_images()