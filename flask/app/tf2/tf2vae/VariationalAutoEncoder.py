import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
import pandas as pd
import pickle
import tensorflow_probability as tfp
ds = tfp.distributions
import sys 
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
from utils import read_image_list_file, preprocess_image, load_and_preprocess_image


class VAE(tf.keras.Model):
    """a basic vae class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def reconstruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu)

    def decode(self, z):
        return self.dec(z)

    def compute_loss(self, x):

        q_z = self.encode(x)
        z = q_z.sample()
        x_recon = self.decode(z)
        p_z = ds.MultivariateNormalDiag(
          loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
          )
        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
#         recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - x_recon), axis=0))
        recon_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(x - x_recon), axis=0)/ (2*tf.math.square(0.1)) + tf.math.log(0.1))
        return recon_loss, latent_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

def architecture(dims, n_z):
    """
    The architecture of the model
    """
    encoder = [
        tf.keras.layers.InputLayer(input_shape=dims),
        tf.keras.layers.Conv2D(
            filters=8, kernel_size=3, strides=(1, 1), activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=(1, 1), activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=(1, 1), activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(1, 1), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=n_z*2),
    ]

    decoder = [
        tf.keras.layers.Dense(units=32 * 16 * 32, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(32, 16, 32)),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
        ),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2DTranspose(
            filters=16, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
        ),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2DTranspose(
            filters=16, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
        ),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        )
    ]
    return encoder, decoder

def training():
    file = open('../localdata/train_list', 'rb')
    train_list = pickle.load(file)
    file.close()
    file = open('../localdata/test_list', 'rb')
    test_list = pickle.load(file)
    file.close()

    TRAIN_BUF=len(train_list)
    BATCH_SIZE=64
    TEST_BUF=len(test_list)
    DIMS = (256,128,3)
    N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    N_Z = 64   #latent space dimension
    
    train_path_ds = tf.data.Dataset.from_tensor_slices(train_list)
    train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    train_image_ds = train_image_ds.repeat()
    train_image_ds = train_image_ds.batch(BATCH_SIZE)
    train_dataset = train_image_ds.prefetch(buffer_size=AUTOTUNE)

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_list)
    test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.repeat()
    test_image_ds = test_image_ds.batch(BATCH_SIZE)
    test_dataset = test_image_ds.prefetch(buffer_size=AUTOTUNE)

    # define encoder and decoder
    encoder, decoder = architecture(DIMS, N_Z)
    # the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(3e-4)
    # train the model
    model = VAE(
        enc = encoder,
        dec = decoder,
        optimizer = optimizer,
    )
    example_data = next(iter(test_dataset))
    model.train(example_data)
    n_epochs = 201
    start_epoch = 26
    model.load_weights(filepath='checkpoints/weight_{}.h5'.format(start_epoch))
    # a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns = ['recon_loss', 'latent_loss'])
    step = 0
    for epoch in range(start_epoch+1, n_epochs):
        shuffled_train_list = random.sample(train_list, len(train_list))
        train_path_ds = tf.data.Dataset.from_tensor_slices(shuffled_train_list)
        train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        train_image_ds = train_image_ds.repeat()
        train_image_ds = train_image_ds.batch(BATCH_SIZE)
        train_dataset = train_image_ds.prefetch(buffer_size=AUTOTUNE)
        # train
        for batch, train_x in tqdm(
            zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
        ):
            step = step +1
            model.train(train_x)
            train_recon_loss, train_latent_loss = model.compute_loss(train_x)
            if step % 10 ==0:
                print(
                    "Step: {} | recon_loss: {} | latent_loss: {}".format(
                    step, train_recon_loss, train_latent_loss)
                )
        # test on holdout
        shuffled_test_list = random.sample(test_list, len(test_list))
        test_path_ds = tf.data.Dataset.from_tensor_slices(shuffled_test_list)
        test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        test_image_ds = test_image_ds.repeat()
        test_image_ds = test_image_ds.batch(BATCH_SIZE)
        test_dataset = test_image_ds.prefetch(buffer_size=AUTOTUNE)
        loss = []
        for batch, test_x in tqdm(
            zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
        ):
            loss.append(model.compute_loss(test_x))
        losses.loc[len(losses)] = np.mean(loss, axis=0)
#         # plot results
#         display.clear_output()
        print(
            "Epoch: {} | recon_loss: {} | latent_loss: {}".format(
                epoch, losses.recon_loss.values[-1], losses.latent_loss.values[-1]
            )
        )
    
def save_bottleneck(start_epoch):
    file = open('../localdata/train_list', 'rb')
    train_list = pickle.load(file)
    file.close()
    TRAIN_BUF=len(train_list)
    BATCH_SIZE=64
    DIMS = (256,128,3)
    N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    N_Z = 64   #latent space dimension
    
    train_path_ds = tf.data.Dataset.from_tensor_slices(train_list)
    train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    train_image_ds = train_image_ds.repeat()
    train_image_ds = train_image_ds.batch(BATCH_SIZE)
    train_dataset = train_image_ds.prefetch(buffer_size=AUTOTUNE)
    
    # create an numpy array to store the latent space
    ui_vectors = np.zeros(shape=(0,N_Z))
    # define encoder and decoder
    encoder, decoder = architecture(DIMS, N_Z)
    # the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(3e-4)
    # train the model
    model = VAE(
        enc = encoder,
        dec = decoder,
        optimizer = optimizer,
    )
    example_data = next(iter(train_dataset))
    model.train(example_data)
    model.load_weights(filepath='checkpoints/weight_{}.h5'.format(start_epoch))
    # a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns = ['recon_loss', 'latent_loss'])
    step = 0
    # iterate only once to save the latent vectors
    for epoch in range(start_epoch+1, start_epoch+2):
        # train
        for batch, train_x in tqdm(
            zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
        ):
            step = step +1
            q_z = model.encode(train_x)
            z = q_z.sample()
            z_np = z.numpy()
            ui_vectors = np.append(ui_vectors, z_np, axis = 0)
    np.save("ui_vectors.npy", ui_vectors)
    
def preloading(pre_trained_weights, image):
#     file = open('{}/app/tf2/test_list'.format(os.path.abspath(os.getcwd())), 'rb')
#     test_list = pickle.load(file)
#     file.close()
    BATCH_SIZE=64
    DIMS = (256,128,3)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    N_Z = 64   #latent space dimension
    # define encoder and decoder
    encoder, decoder = architecture(DIMS, N_Z)
    # the optimizer for the model
    optimizer = tf.keras.optimizers.Adam(3e-4)
    # build the model
    model = VAE(
        enc = encoder,
        dec = decoder,
        optimizer = optimizer)
    image_prep = load_and_preprocess_image(image)
    img = tf.expand_dims(image_prep, 0)
    model.train(img)
    model.load_weights(filepath=pre_trained_weights)
    return model
    
def test(model, image):
    # preprocess the image, the final format should be Tensor(256,128,3)
    image_prep = load_and_preprocess_image(image)
    img = tf.expand_dims(image_prep, 0)
#     image_tf = tf.data.Dataset.from_tensor_slices(image_prep)
#     img = next(iter(image_tf))
    q_z = model.encode(img)
    z = q_z.sample()
#     z_walk = z + 0.5 * np.ones_like(z)
#     img_reconstructed = model.decode(z)
#     img_generated = model.decode(z_walk)
    return image_prep, z

def generated_encode(model, image):
    img = tf.expand_dims(image, 0)
    q_z = model.encode(img)
    z = q_z.sample()
    return z

def generate(model, z):
    return model.decode(z)