import sys
import time

from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import backend as K
from keras.models import load_model
K.set_image_data_format('channels_last')
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2


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


def define_model_1():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Batch NORM, increasing dropout
def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def summarize_diagnostics_1(history, history1):
    pyplot.figure(figsize=(8, 6))
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train-with-batchnorm-lr-0.1')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test-with-batchnorm-lr-0.1')
    pyplot.plot(history1.history['accuracy'], color='green', label='train-without-batchnorm-lr-0.001')
    pyplot.plot(history1.history['val_accuracy'], color='black', label='test-without-batchnorm-lr-0.001')
    pyplot.legend(loc="best")
    pyplot.savefig('better_performance.png')
    pyplot.close()

def summarize_diagnostics_2(history, history_train,history1, history_train1):
    pyplot.figure(figsize=(8,6))
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train-with-batchnorm-lr-0.001')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test-with-batchnorm-lr-0.001')
    pyplot.plot(history_train.history['val_accuracy'], color='red', label='train-as-test-with-batchnorm-lr-0.001')
    pyplot.plot(history1.history['accuracy'], color='green', label='train-without-batchnorm-lr-0.001')
    pyplot.plot(history1.history['val_accuracy'], color='black', label='test-without-batchnorm-lr-0.001')
    # pyplot.plot(history_train1.history['accuracy'], color='yellow', label='train-without-batchnorm-lr-0.001')
    pyplot.plot(history_train1.history['val_accuracy'], color='yellow', label='train-as-test-without-batchnorm-lr-0.001')
    pyplot.legend(loc="best")
    pyplot.savefig('accelerate_convergence_epochs_50.png')
    pyplot.close()


def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    model_1 = define_model_1()
    # datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # it_train = datagen.flow(trainX, trainY, batch_size=64)
    # steps = int(trainX.shape[0] / 64)
    # fit model
    history = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=0)
    history_train = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(trainX, trainY), verbose=0)
    history_1 = model_1.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=0)
    history_train_1 = model_1.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(trainX, trainY), verbose=0)
    # history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=400, validation_data=(testX, testY), verbose=0)
    model.save('final_model_50_epochs_with_accelerate.h5')
    model_1.save('final_model_50_epochs_without_accelerate.h5')
    # history_1 = model_1.fit_generator(it_train, steps_per_epoch=steps, epochs=400, validation_data=(testX, testY), verbose=0)
    # evaluate model
    loss, acc = model.evaluate(testX, testY, verbose=0)
    loss_1, acc_1 = model_1.evaluate(testX, testY, verbose=0)
    print('The accuracy without Batch Norm is:', (acc_1 * 100.0))
    print('The loss without Batch Norm is:', loss_1)
    print('The accuracy with Batch Norm is:', (acc * 100.0))
    print('The loss with Batch Norm is:', loss)
    # learning curves
    # summarize_diagnostics_1(history, history_1)
    summarize_diagnostics_2(history,history_train, history_1, history_train_1)


def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img


def visualize_filters():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    layers = model.layers
    layer_ids = [0, 6, 12]
    fig, ax = pyplot.subplots(nrows=1, ncols=3)
    for i in range(3):
        ax[i].imshow(layers[layer_ids[i]].get_weights()[0][:, :, :, 0][:, :, 0], cmap='gray')
        ax[i].set_title('block' + str(i + 1))
        ax[i].set_xticks([])
        ax[i].set_yticks([])


def heat_maps():
    model = load_model('./final_model_new_data-400epochs.h5')
    x = load_image(img_loc)
    # preds = model.predict(x)
    result = model.predict_classes(x)[0]
    # print(preds)
    print(result)
    pred_vector_output = model.output[:, result]
    heatmap = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
            continue
        some_conv_layer = layer.output
        grads = K.gradients(pred_vector_output, some_conv_layer)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        iterate = K.function([model.input], [pooled_grads, some_conv_layer[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for j in range(some_conv_layer.shape[-1]):
            conv_layer_output_value[:, :, j] *= pooled_grads_value[j]
        temp_out = np.sort(conv_layer_output_value)
        for k in range(1, 4):
            heatmap.append(temp_out[:, :, -k])
    return heatmap


def show_plot_activation(heatmap):
    layer_names = ['conv2d_1_1', 'conv2d_1_2', 'conv2d_1_3', 'conv2d_2_1', 'conv2d_2_2', 'conv2d_2_3',
                   'conv2d_3_1', 'conv2d_3_2', 'conv2d_3_3', 'conv2d_4_1', 'conv2d_4_2', 'conv2d_4_3', 'conv2d_5_1',
                   'conv2d_5_2', 'conv2d_5_3', 'conv2d_6_1', 'conv2d_6_2', 'conv2d_6_3']
    pyplot.figure(figsize=(18, 18))
    for i in range(18):
        pyplot.figure(figsize=(40, 40))
        pyplot.subplot(6, 3, i + 1)
        img_heatmap = np.maximum(heatmap[i], 0)
        img_heatmap /= np.max(img_heatmap)
        pyplot.imshow(img_heatmap)
        pyplot.title(layer_names[i])
        pyplot.show()


def plot_on_input(heatmap):
    img = cv2.imread(img_loc)
    layer_names = ['conv2d_1_1', 'conv2d_1_2', 'conv2d_1_3', 'conv2d_2_1', 'conv2d_2_2', 'conv2d_2_3',
                   'conv2d_3_1', 'conv2d_3_2', 'conv2d_3_3', 'conv2d_4_1', 'conv2d_4_2', 'conv2d_4_3', 'conv2d_5_1',
                   'conv2d_5_2', 'conv2d_5_3', 'conv2d_6_1', 'conv2d_6_2', 'conv2d_6_3']
    for i, hm in enumerate(heatmap):
        img_heatmap = np.maximum(hm, 0)
        img_heatmap /= np.max(img_heatmap)
        img_hm = cv2.resize(img_heatmap, (img.shape[1], img.shape[0]))
        img_hm = np.uint8(255 * img_hm)
        img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_JET)
        superimposed_img = img_hm * 0.4 + img
        cv2.imwrite('./heatmaps/ship_data_aug_400/ship_{}.png'.format(layer_names[i]), superimposed_img)


# entry point, run the test harness
t0 = time.time()
run_test_harness()
t1 = time.time() - t0
print("Time elapsed: ", t1 / 60.0)
# visualize_filters()
img_loc = './ship.png'
# hmp=heat_maps()
# show_plot_activation(hmp)
# plot_on_input(hmp)
# pyplot.figure(figsize=(16, 16))
# layer_name = 'conv2d_1_1'
# img = image.img_to_array(
# image.load_img('deer_{}.png'.format(layer_name))) / 255.
# pyplot.imshow(img)
# pyplot.title(layer_name)
# train_X, train_Y, test_X, test_Y=load_dataset()
# trainX, testX = prep_pixels(train_X, test_X)
# model = load_model('./final_model_50_epochs.h5')
# loss, acc = model.evaluate(testX, test_Y, verbose=0)
# print('The accuracy with Batch Norm is:', (acc * 100.0))
# print('The loss with Batch Norm is:', loss)
    # learning curves
    # summarize_diagnostics_1(history, history_1)
