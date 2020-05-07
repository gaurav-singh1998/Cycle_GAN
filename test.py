from __future__ import print_function, unicode_literals, absolute_import, division
import tensorflow as tf
import os
from utils import preprocess_image_test, generate_images_test
from model import ResnetGenerator
tf.get_logger().setLevel('WARNING')

PATH_test_B='ukiyoe2photo/testB/'
batch_size=1
testB_size = len(os.listdir(PATH_test_B))

test_B=tf.data.Dataset.list_files(PATH_test_B+'*.jpg')
test_B=test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(testB_size).batch(batch_size)
test_B=test_B.prefetch(batch_size)

generator_f = ResnetGenerator()

checkpoint_path_final_model = "./checkpoint/final_model"


ckpt = tf.train.Checkpoint(generator_f=generator_f)

ckpt_manager_final_model = tf.train.CheckpointManager(ckpt, checkpoint_path_final_model, max_to_keep=1)

if ckpt_manager_final_model.latest_checkpoint:
  ckpt.restore(ckpt_manager_final_model.latest_checkpoint)
  print ('Latest checkpoint restored!!')
  for inp in test_B.take(5):
    generate_images_test(generator_f, inp)

else:
  print("Download the pretrained model from \'https://drive.google.com/drive/folders/1-tKDLeaRJ_5Kn3gkWAvZh6FcgeNK8mWV?usp=sharing' else train the model from scratch by running train.py")
