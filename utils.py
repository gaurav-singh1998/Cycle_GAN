from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = tf.io.read_file(image)
  image = tf.image.decode_jpeg(image, channels=3)
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = tf.io.read_file(image)
  image = tf.image.decode_image(image, channels=3)
  image = normalize(image)
  return image

def generate_images_train(model_A, test_input_A, model_B, test_input_B, epoch):
  prediction_A = model_A(test_input_A)
  prediction_B = model_B(test_input_B)
  
  plt.figure(figsize=(12, 12))
  display_list_A = [test_input_A[0], prediction_A[0]]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list_A[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig('generated(A->B)_{}.png'.format(epoch+1))
  plt.close()

  plt.figure(figsize=(12, 12))
  display_list_B = [test_input_B[0], prediction_B[0]]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list_B[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig('generated(B->A)_{}.png'.format(epoch+1))
  plt.close()

  os.rename('./generated(A->B)_{}.png'.format(epoch+1), './pictures/generated(A->B)_{}.png'.format(epoch+1))
  os.rename('./generated(B->A)_{}.png'.format(epoch+1), './pictures/generated(B->A)_{}.png'.format(epoch+1))


def generate_images_test(model, test_input):
  prediction = model(test_input)
  
  plt.figure(figsize=(12, 12))
  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  plt.close()

