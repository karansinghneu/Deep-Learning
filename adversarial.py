import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
import copy

def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def adversarial_pattern(image, target_label, model):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        print('The original predicted class is', np.argmax(prediction))
        loss = tf.keras.losses.categorical_crossentropy(target_label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)

    return signed_grad

def run_targeted_attack(idx_1, idx_2):
  trainX, trainY, testX, testY = load_dataset()
  trainX, testX = prep_pixels(trainX, testX)
  img = testX[idx_1]
  true_class = testY[idx_1]
  image_label = testY[idx_2]
  target_img = testX[idx_2]
  image_label = np.reshape(image_label, (1,10))
  img_reshaped = img.reshape((1, 32, 32, 3))
  model = tf.keras.models.load_model('tf_model.h5')
  perturbations = adversarial_pattern(img_reshaped, image_label,model).numpy()
  # perturbations = np.clip(perturbations,0.,1.)
  adversarial = img_reshaped - perturbations * 0.01
  adversarial = np.clip(adversarial,0.,1.)
  perturbations = np.clip(perturbations,0.,1.)
  adv_prediction = model(adversarial)
  print('The true class is',np.argmax(true_class))
  print('The target class is',np.argmax(image_label))
  print('The adversarial predicted class is',np.argmax(adv_prediction))
  adv_image = adversarial.reshape((32,32,3))
  perturb_plot = perturbations.reshape((32,32,3))
  plt.figure(figsize=(18, 18))
  plt.subplot(330 + 1 + 0)
  plt.imshow(img)
  plt.title('Original image with true and predicted class: 0')

  plt.subplot(330 + 1 + 1)
  plt.imshow(perturb_plot)
  plt.title('Perturbation with target class: 5')

  plt.subplot(330 + 1 + 2)
  plt.imshow(adv_image)
  plt.title('Perturbed image with predicted class: 5')
  # plt.subplot(340 + 1 + 3)
  # plt.imshow(target_img)
  # plt.title('Classifiers output')
  # plt.savefig('sample_from_test.png')
  plt.show()

def deepfool(image_norm, model, num_classes=10, overshoot=0.02, max_iter=50):

  f_image = model(image_norm).numpy().flatten()
  I = (np.array(f_image)).flatten().argsort()[::-1]
  I = I[0:num_classes]
  label = I[0]

  input_shape = np.shape(image_norm)
  pert_image = copy.deepcopy(image_norm)
  w = np.zeros(input_shape)
  r_tot = np.zeros(input_shape)

  loop_i = 0
  x = tf.Variable(pert_image)
  fs = model(x)  # shape=(1, num_classes)
  k_i = label

  def loss_func(logits, I, k):
    return logits[0, I[k]]

  while k_i == label and loop_i < max_iter:
    pert = np.inf
    one_hot_label_0 = tf.one_hot(label, num_classes)
    with tf.GradientTape() as tape:
      tape.watch(x)
      fs = model(x)
      loss_value = loss_func(fs, I, 0)
    grad_orig = tape.gradient(loss_value, x)

    for k in range(1, num_classes):
      one_hot_label_k = tf.one_hot(I[k], num_classes)
      with tf.GradientTape() as tape:
        tape.watch(x)
        fs = model(x)
        loss_value = loss_func(fs, I, k)
      cur_grad = tape.gradient(loss_value, x)

      w_k = cur_grad - grad_orig

      f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

      pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))

      if pert_k < pert:
        pert = pert_k
        w = w_k

    r_i = (pert + 1e-4) * w / np.linalg.norm(w)
    r_tot = np.float32(r_tot + r_i)
    pert_image = image_norm + (1 + overshoot) * r_tot

    x = tf.Variable(pert_image)

    fs = model(x)
    k_i = np.argmax(np.array(fs).flatten())

    loop_i += 1

  r_tot = (1 + overshoot) * r_tot

  return r_tot, loop_i, label, k_i, pert_image


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    if p == 2:
        v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset, f, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02,
                           max_iter_df=10):
    v = 0
    fooling_rate = 0.0
    num_images = np.shape(dataset)[0]

    itr = 0
    while fooling_rate < 1 - delta and itr < max_iter_uni:

        np.random.shuffle(dataset)

        print('Starting pass number ', itr)

        for k in range(0, num_images):
            cur_img = dataset[k:(k + 1), :, :, :]

            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img + v)).flatten())):
                if k % 5000 == 0:
                    print('>> k = ', k, ', pass #', itr)

                    dr, iter, _, _, _ = deepfool(cur_img + v, f, num_classes=num_classes, overshoot=overshoot,
                                                 max_iter=max_iter_df)

                    if iter < max_iter_df - 1:
                        v = v + dr

                        v = proj_lp(v, xi, p)

        itr = itr + 1

        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii + 1) * batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)

    return v


def universal_attack():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)

    model = tf.keras.models.load_model('tf_model.h5')
    universal_pert = universal_perturbation(trainX, model)
    np.save('univ_pert_latest.npy', universal_pert)
    print(universal_pert.shape)


def execute_attack():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    idx = 61
    # 1008 => works
    sample_image_1 = trainX[idx]
    sample_image_1_new = sample_image_1.reshape((1, 32, 32, 3))

    print('The ground truth class is:', np.argmax(trainY[idx]))
    universal_pert = np.load('universal_perturbation_2.npy')
    model = tf.keras.models.load_model('tf_model.h5')

    true_output = model(sample_image_1_new)
    true_output_class = np.argmax(true_output)
    print('The predicted output class before attack is:', true_output_class)
    universal_pert = np.clip(universal_pert, 0., 1.)
    sample_image_1_perturbed = sample_image_1_new + universal_pert * 0.23

    sample_image_1_perturbed = np.clip(sample_image_1_perturbed, 0., 1.)
    predicted_output = model(sample_image_1_perturbed)
    predicted_output_class = np.argmax(predicted_output)
    print('The predicted output class after attack is:', predicted_output_class)
    # universal_pert = np.clip(universal_pert,0.,1.)
    adv_image = sample_image_1_perturbed.reshape((32, 32, 3))
    perturb_plot = universal_pert.reshape((32, 32, 3))

    plt.figure(figsize=(18, 18))
    plt.subplot(330 + 1 + 0)
    plt.imshow(sample_image_1)
    plt.title('Original image with true and predicted class: 1')
    plt.subplot(330 + 1 + 1)
    plt.imshow(perturb_plot)
    plt.title('Universal Perturbation')
    plt.subplot(330 + 1 + 2)
    plt.imshow(adv_image)
    plt.title('Perturbed Image with predicted class 9')

    plt.show()



run_targeted_attack(10,12)
# (10,100)--> Works
# (20,200)--> Works
# (20,10) -- > Works
# (10,12)--> Works
universal_attack()
execute_attack()