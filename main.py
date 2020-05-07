from __future__ import print_function, unicode_literals, absolute_import, division
import argparse
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from PIL import Image
from utils import normalize
from model import ResnetGenerator

parser = argparse.ArgumentParser(description='Transforming the input image to ukiyoe painting style image.')
parser.add_argument('--input_path', type=str, help='Give the path of the source image to be converted including the name of the source image.')
parser.add_argument('--output_path', type=str, help='Give the path of the directory in which you wish to store the converted image including the name of the image.')
args = parser.parse_args()


def array_image_save(array, image_path =args.output_path):
  image.save(image_path)
  print("Saved image: {}".format(image_path))

og_image = Image.open(args.input_path)
og_image_size = og_image.size
new_width = og_image_size[0]
new_height = og_image_size[1]

image = load_img(args.input_path, target_size=(256, 256))
image = img_to_array(image)
image = normalize(image)
image = image[tf.newaxis, ...]

generator_f=ResnetGenerator()

checkpoint_path_final_model = "./checkpoint/final_model"


ckpt = tf.train.Checkpoint(generator_f=generator_f)

ckpt_manager_final_model = tf.train.CheckpointManager(ckpt, checkpoint_path_final_model, max_to_keep=1)

if ckpt_manager_final_model.latest_checkpoint:
  ckpt.restore(ckpt_manager_final_model.latest_checkpoint)
  print ('Latest checkpoint restored!!')
  image=generator_f(image)
  image=tf.reshape(image, (256, 256, 3))
  image=array_to_img(image)
  image = image.resize((new_width, new_height), Image.ANTIALIAS)
  array_image_save(image)

else:
  print("Download the pretrained model from \'https://drive.google.com/drive/folders/1-tKDLeaRJ_5Kn3gkWAvZh6FcgeNK8mWV?usp=sharing' else train the model from scratch by running train.py")
