from keras.utils import to_categorical
from keras.datasets import cifar10, mnist
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import keras.backend as K


def split_dataset(dataset, output):
    datasetA = []
    outputA = []
    datasetB = []
    outputB = []
    for i in range(0, len(output)):
        if output[i] in range(0, 5):
            datasetA.append(dataset[i])
            outputA.append(output[i])
        else:
            datasetB.append(dataset[i])
            outputB.append(output[i])

    taskA_dataset = np.stack(datasetA)
    taskA_output = np.stack(outputA)
    taskB_dataset = np.stack(datasetB)
    taskB_output = np.stack(outputB)
    taskA_output = to_categorical(taskA_output, num_classes=10)
    taskB_output = to_categorical(taskB_output)
    return taskA_dataset, taskA_output, taskB_dataset, taskB_output


def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    taskA_input_train, taskA_output_train, taskB_input_train, taskB_output_train = split_dataset(trainX, trainY)
    taskA_input_test, taskA_output_test, taskB_input_test, taskB_output_test = split_dataset(testX, testY)

    return taskA_input_train, taskA_output_train, taskB_input_train, taskB_output_train, taskA_input_test, taskA_output_test, taskB_input_test, taskB_output_test


def MLP():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                    bias_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                    bias_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                    bias_regularizer=tf.keras.regularizers.l1(0.001)))
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history_first, history_second, history_additional, history_additional_new):
    task_A_result = history_first.history['val_accuracy'] + history_additional.history['val2_accuracy']

    task_B_result = history_second.history['val_accuracy'] + history_additional_new.history['val3_accuracy']

    print(task_A_result)
    print(len(task_A_result))

    print(task_B_result)
    print(len(task_B_result))

    with open('mlp_mnist_A', 'wb') as f:
        pickle.dump(task_A_result, f)
    with open('mlp_mnist_B', 'wb') as f:
        pickle.dump(task_B_result, f)


class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    # print(self.model.metrics[i].name)
                    valuename = validation_set_name + '_' + self.model.metrics[i].name
                self.history.setdefault(valuename, []).append(result)


def run_test_harness():
    taskA_input_train, taskA_output_train, taskB_input_train, taskB_output_train, taskA_input_test, taskA_output_test, taskB_input_test, taskB_output_test = load_dataset()
    taskA_input_train, taskA_input_test = prep_pixels(taskA_input_train, taskA_input_test)
    taskB_input_train, taskB_input_test = prep_pixels(taskB_input_train, taskB_input_test)

    # model = define_model()
    model = MLP()
    history_additional = AdditionalValidationSets([(taskB_input_test, taskB_output_test, 'val2')])
    history = model.fit(taskA_input_train, taskA_output_train, epochs=20, batch_size=64,
                        validation_data=(taskA_input_test, taskA_output_test), verbose=0,
                        callbacks=[history_additional])
    tf.keras.models.save_model(model=model, filepath='mlp_mnist_A_model.h5')

    history_additional_new = AdditionalValidationSets([(taskB_input_test, taskB_output_test, 'val3')])
    model_new = MLP()
    model_new.set_weights(model.get_weights())
    history_new = model_new.fit(taskB_input_train, taskB_output_train, epochs=20, batch_size=64,
                                validation_data=(taskA_input_test, taskA_output_test), verbose=0,
                                callbacks=[history_additional_new])
    tf.keras.models.save_model(model=model_new, filepath='mlp_mnist_B_model.h5')

    summarize_diagnostics(history, history_new, history_additional, history_additional_new)


def computer_fisher(model, imgset, num_sample=100):
    f_accum = []
    for i in range(len(model.weights)):
        f_accum.append(np.zeros(K.int_shape(model.weights[i])))
    f_accum = np.array(f_accum)
    for j in range(num_sample):
        print('Sample Iteration: ', j)
        img_index = np.random.randint(imgset.shape[0])
        for m in range(len(model.weights)):
            grads = K.gradients(K.log(model.output), model.weights)[m]
            result = K.function([model.input], [grads])
            f_accum[m] += np.square(result([np.expand_dims(imgset[img_index], 0)])[0])
    f_accum /= num_sample
    return f_accum


class ewc_reg(tf.keras.regularizers.Regularizer):
    def __init__(self, fisher, prior_weights, Lambda=0.2):
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def __call__(self, x):
        regularization = 0.
        regularization += self.Lambda * K.sum(
            self.fisher * K.square(x - self.prior_weights)) + 0.001 * tf.math.reduce_sum(tf.math.abs(x))
        # regularization +=  self.Lambda * K.sum(self.fisher * K.square(x - self.prior_weights))
        return regularization

    def get_config(self):
        return {'Lambda': float(self.Lambda)}


def ewc_training():
    taskA_input_train, taskA_output_train, taskB_input_train, taskB_output_train, taskA_input_test, taskA_output_test, taskB_input_test, taskB_output_test = load_dataset()
    taskA_input_train, taskA_input_test = prep_pixels(taskA_input_train, taskA_input_test)
    taskB_input_train, taskB_input_test = prep_pixels(taskB_input_train, taskB_input_test)

    model_old = MLP()
    history_additional = AdditionalValidationSets([(taskB_input_test, taskB_output_test, 'val2')])
    history = model_old.fit(taskA_input_train, taskA_output_train, epochs=20, batch_size=64,
                            validation_data=(taskA_input_test, taskA_output_test), verbose=0,
                            callbacks=[history_additional])
    tf.keras.models.save_model(model=model_old, filepath='first_task_mlp_ewc.h5')
    task_A_result = history.history['val_accuracy'] + history_additional.history['val2_accuracy']

    with open('first_task_mlp_ewc_result', 'wb') as f:
        pickle.dump(task_A_result, f)

    I = computer_fisher(model_old, taskA_input_train)

    model_new = tf.keras.Sequential()
    model_new.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model_new.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=ewc_reg(I[0], model_old.weights[0]),
                                        bias_regularizer=ewc_reg(I[1], model_old.weights[1])))
    model_new.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=ewc_reg(I[2], model_old.weights[2]),
                                        bias_regularizer=ewc_reg(I[3], model_old.weights[3])))
    # model_new.add(tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=ewc_reg(I[4], model_old.weights[4]),
    #                 bias_regularizer=ewc_reg(I[5], model_old.weights[5])))
    model_new.add(
        tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=ewc_reg(I[4], model_old.weights[4]),
                              bias_regularizer=ewc_reg(I[5], model_old.weights[5])))
    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    # opt =	tf.keras.optimizers.Adam()
    model_new.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history_additional_new = AdditionalValidationSets([(taskB_input_test, taskB_output_test, 'val3')])
    model_new.set_weights(model_old.get_weights())
    history_new = model_new.fit(taskB_input_train, taskB_output_train, epochs=20, batch_size=64,
                                validation_data=(taskA_input_test, taskA_output_test), verbose=0,
                                callbacks=[history_additional_new])
    tf.keras.models.save_model(model=model_new, filepath='second_task_mlp_ewc.h5')

    task_B_result_ewc = history_new.history['val_accuracy'] + history_additional_new.history['val3_accuracy']

    with open('second_task_mlp_ewc_result', 'wb') as f:
        pickle.dump(task_B_result_ewc, f)

    print(len(I))
    print(I.shape)


def generate_plots():
    with open('first_task_mlp_ewc_result', 'rb') as f:
        task_A_result = pickle.load(f)
    with open('second_task_mlp_ewc_result', 'rb') as f:
        task_B_result = pickle.load(f)

    # plt.figure(figsize=(10,8))
    f, (ax1, ax2) = plt.subplots(1, 2,
                                 sharey=True, figsize=(8, 6))
    # derivative = 2 * x * np.cos(x**2) - np.sin(x)
    ax1.plot([(1 - x) for x in task_A_result[0:20]], color='orange', label='Test-A')
    ax1.plot([(1 - x) for x in task_A_result[20:40]], color='black', label='Test-B')
    ax2.plot([(1 - x) for x in task_B_result[0:20]], color='orange', label='Test-A')
    ax2.plot([(1 - x) for x in task_B_result[20:40]], color='black', label='Test-B')

    ax1.title.set_text('Train-A')
    ax2.title.set_text('Train-B')

    plt.suptitle('Classification Error')

    plt.legend(loc="best")
    f.subplots_adjust(wspace=0)
    plt.savefig('mlp_mnist_ewc.png')


# entry point, run the test harness
t0 = time.time()
run_test_harness()
t1 = time.time() - t0
print("Time elapsed: ", t1 / 60.0)
ewc_training()
generate_plots()
