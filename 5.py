# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Load Dataset (CelebA)
def load_celeba(csv_path, image_dir, img_size=(64, 64), batch_size=32):
    # Read CSV file
    df = pd.read_csv(csv_path)
    image_paths = [os.path.join(image_dir, img_id) for img_id in df['image_id']]
    
    # Create TensorFlow Dataset
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        return image
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Define Generator
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((8, 8, 256)),
        
        layers.Conv2DTranspose(128, (4,4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(64, (4,4), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(3, (4,4), strides=1, padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Define Discriminator
def build_discriminator(img_shape):
    model = models.Sequential([
        layers.Conv2D(64, (4,4), strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(128, (4,4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(256, (4,4), strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Training Parameters
latent_dim = 100
img_shape = (64, 64, 3)
batch_size = 32

# Build Models
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Compile Discriminator
discriminator.compile(
    optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build and Compile GAN
discriminator.trainable = False
gan = models.Sequential([generator, discriminator])
gan.compile(
    optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# Training Functions
def save_imgs(epoch, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r*c, latent_dim))
    gen_imgs = generator.predict(noise)
    
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"gan_images_{epoch}.png")
    plt.close()

def train_gan(dataset, epochs=100, save_interval=10):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        start = time.time()
        
        for batch in dataset:
            # Train Discriminator
            noise = tf.random.normal([batch_size, latent_dim])
            gen_imgs = generator(noise, training=False)
            
            d_loss_real = discriminator.train_on_batch(batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = tf.random.normal([batch_size, latent_dim])
            g_loss = gan.train_on_batch(noise, real_labels)
        
        if epoch % save_interval == 0:
            save_imgs(epoch, generator)
            print(f"Epoch {epoch} [D loss: {d_loss[0]} | D acc: {100*d_loss[1]}] [G loss: {g_loss}] Time: {time.time()-start:.2f}s")

# Load Dataset
csv_path = "list_attr_celeba.csv"  # Update with your CSV path
image_dir = "img_align_celeba"     # Update with your image directory
dataset = load_celeba(csv_path, image_dir)

# Start Training
train_gan(dataset, epochs=100, save_interval=10)