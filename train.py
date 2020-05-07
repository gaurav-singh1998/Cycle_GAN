from __future__ import print_function, unicode_literals, absolute_import, division
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import time
import os
from utils import preprocess_image_train, generate_images_train
from model import ResnetGenerator, ConvDiscriminator, LinearDecay
os.mkdir('./pictures')
PATH_train_A='ukiyoe2photo/trainA/'
PATH_train_B='ukiyoe2photo/trainB/'
EPOCHS = 200
trainA_size = len(os.listdir(PATH_train_A))
trainB_size = len(os.listdir(PATH_train_B))
batch_size=1 #Change if multi gpu
len_dataset = max(trainA_size, trainB_size) // batch_size

print('Building data input pipeline.....')
train_A=tf.data.Dataset.list_files(PATH_train_A+'*.jpg')
train_A=train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(trainA_size).batch(batch_size)
train_A=train_A.prefetch(batch_size)

train_B=tf.data.Dataset.list_files(PATH_train_B+'*.jpg')
train_B=train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(trainB_size).batch(batch_size)
train_B=train_B.prefetch(batch_size)
print('Done!!!')

sample_ukiyoe = next(iter(train_A))
sample_photo = next(iter(train_B))

generator_g=ResnetGenerator()
generator_f=ResnetGenerator()

discriminator_x=ConvDiscriminator()
discriminator_y=ConvDiscriminator()

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

G_lr_scheduler = LinearDecay(0.0002, EPOCHS * len_dataset, 100 * len_dataset)
D_lr_scheduler = LinearDecay(0.0002, EPOCHS * len_dataset, 100 * len_dataset)

generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=0.5)



checkpoint_path = "./checkpoints/train"
checkpoint_path_final_model = "./checkpoint/final_model"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
ckpt_manager_final_model = tf.train.CheckpointManager(ckpt, checkpoint_path_final_model, max_to_keep=1)


if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x)
    cycled_x = generator_f(fake_y)

    fake_x = generator_f(real_y)
    cycled_y = generator_g(fake_x)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x)
    same_y = generator_g(real_y)

    disc_real_x = discriminator_x(real_x)
    disc_real_y = discriminator_y(real_y)

    disc_fake_x = discriminator_x(fake_x)
    disc_fake_y = discriminator_y(fake_y)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


if __name__ == '__main__':
  print("Training loop started\n")
  for epoch in range(EPOCHS):
    start = time.time()
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_A, train_B)):
      train_step(image_x, image_y)
      if n%10==0:
        print ('.', end='')
      n+=1
    print('\n')

    generate_images_train(generator_g, sample_ukiyoe, generator_f, sample_photo, epoch)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    if (epoch+1)==200:
        ckpt_save_path=ckpt_manager_final_model.save()
        print("Final model saved at {}".format(ckpt_save_path))

    print ('Time taken for epoch {} is {} minutes\n'.format(epoch + 1,
                                                      (int(time.time()-start)/60)))
  print("Done!!!\n")
