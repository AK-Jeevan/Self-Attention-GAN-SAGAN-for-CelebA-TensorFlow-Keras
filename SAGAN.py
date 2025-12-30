"""
Self-Attention Generative Adversarial Network (SAGAN) – TensorFlow Implementation

This script implements a full Self-Attention GAN (SAGAN) using TensorFlow and Keras.
The model is trained on the CelebA dataset to generate realistic human face images.

Key Components:
- Generator with transposed convolutions and self-attention layers
- Discriminator with spectral normalization and self-attention
- Self-attention mechanism to capture long-range spatial dependencies
- Learnable gamma parameter to safely control attention contribution
- Hinge loss for stable GAN training
- Spectral normalization applied to the discriminator for stability

Architecture Highlights:
- Self-attention allows each spatial position to attend to all others
- Gamma initialized to zero enables gradual learning of attention
- Spectral normalization prevents discriminator overpowering
- Training follows the SAGAN / BigGAN methodology

Dataset:
- CelebA (aligned face images)
- Images resized to 64×64 and normalized to [-1, 1]

Optimizer:
- Adam optimizer (β1 = 0.0, β2 = 0.9)
"""

import tensorflow as tf                      # Main ML library used here.
import tensorflow_datasets as tfds           # Dataset helper with many prebuilt datasets.
import matplotlib.pyplot as plt              # Plotting library (not used heavily here).

# ===============================
# CONFIG
# ===============================
# Settings that control image size, batch size, latent vector size, and epochs.
IMG_SIZE = 64                                # Output image height and width (square).
BATCH_SIZE = 64                              # How many images we process at once.
LATENT_DIM = 128                             # Size of the random vector given to the generator.
EPOCHS = 5                                   # How many passes over the whole dataset.

# ===============================
# DATASET (CelebA)
# ===============================
# Load the CelebA dataset (faces) from tensorflow_datasets.
# tfds.load returns a tf.data.Dataset (or dict of splits); here we get the "train" split.
dataset = tfds.load("celeb_a", split="train")

# Preprocessing function to prepare each example:
# - Resize to IMG_SIZE
# - Convert pixels to float32
# - Scale from [0,255] to [-1, 1] because generator uses tanh at the output.
def preprocess(example):
    # example["image"] is the image tensor from the dataset.
    img = tf.image.resize(example["image"], (IMG_SIZE, IMG_SIZE))
    # tf.cast converts data type; here to float32 so math works as expected.
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img

# Build an efficient dataset pipeline:
# .map - apply preprocess to each example in parallel
# .shuffle - shuffle examples so batches are randomized
# .batch - group examples into batches
# .prefetch - prepare data for the model while the model is training (keeps GPU busy)
dataset = (
    dataset
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # map applies function to each item
    .shuffle(10000)                                        # good practice to shuffle a buffer
    .batch(BATCH_SIZE)                                     # group items into batches
    .prefetch(tf.data.AUTOTUNE)                            # parallelize fetching/prefetching
)

# ===============================
# SPECTRAL NORMALIZATION
# ===============================
# Spectral normalization stabilizes the discriminator by constraining the largest singular value
# of weight matrices. We wrap layers and normalize their kernels before use.
class SpectralNorm(tf.keras.layers.Wrapper):
    # Wrapper: takes any layer (usually Conv2D or Dense) and applies spectral normalisation.
    def __init__(self, layer, power_iterations=1):
        super().__init__(layer)
        # power_iterations controls how many steps of power iteration to approximate spectral norm.
        self.power_iterations = power_iterations

    def build(self, input_shape):
        # Build the wrapped layer (so it creates its variables like kernel)
        self.layer.build(input_shape)
        # Access the kernel (weights) of the wrapped layer.
        self.w = self.layer.kernel
        # Convert shape to python list for indexing/usage later.
        self.w_shape = self.w.shape.as_list()

        # Create a vector `u` used by power iteration. We store it as a non-trainable variable.
        # add_weight creates a new variable; trainable=False means optimizer won't update it.
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer="random_normal",
            trainable=False,
            name="sn_u"
        )

    def call(self, inputs):
        # Reshape w to a 2D matrix with shape [N, out_channels] for power iteration.
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        # Power iteration: approximate the first singular vector/value.
        # tf.matmul performs matrix multiplication.
        # tf.math.l2_normalize divides by L2 norm to keep vectors at unit length.
        for _ in range(self.power_iterations):
            # v = normalize(u * w^T)
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
            # u = normalize(v * w)
            u = tf.math.l2_normalize(tf.matmul(v, w))

        # sigma = v * w * u^T approximates the largest singular value (a scalar).
        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
        # w_norm = w / sigma scales weights by the largest singular value.
        w_norm = w / sigma

        # Assign normalized weights back to the wrapped layer's kernel in original shape.
        # tf.reshape turns w_norm back to the kernel's shape.
        self.layer.kernel.assign(tf.reshape(w_norm, self.w_shape))
        # Keep the updated u for the next call (so we reuse the approximation).
        self.u.assign(u)

        # Finally call the wrapped layer with the normalized kernel and return its output.
        return self.layer(inputs)

# ===============================
# SELF-ATTENTION (SAGAN)
# ===============================
# Self-attention lets layers "look" at other spatial locations to capture long-range dependencies.
# This is the module used in Self-Attention GANs (SAGAN).
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        # query/key/value are 1x1 convolutions that compute projections of the input.
        # channels // 8 reduces dimensionality for query/key to save computation.
        self.query = tf.keras.layers.Conv2D(channels // 8, 1, padding="same")
        self.key   = tf.keras.layers.Conv2D(channels // 8, 1, padding="same")
        self.value = tf.keras.layers.Conv2D(channels, 1, padding="same")
        # gamma is a learned scalar that controls how much attention output contributes.
        self.gamma = self.add_weight(
            shape=(1,), initializer="zeros", trainable=True
        )

    def call(self, x, return_attention=False):
        # Extract dynamic shapes: B=batch, H=height, W=width, C=channels.
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        # Apply the query, key, value convolutions and reshape them to sequences.
        # self.query(x) runs the Conv2D layer on x and returns a tensor.
        # tf.reshape flattens the spatial dimensions H*W into one axis.
        q = tf.reshape(self.query(x), [B, -1, C // 8])   # shape [B, H*W, C/8]
        k = tf.reshape(self.key(x),   [B, -1, C // 8])   # shape [B, H*W, C/8]
        v = tf.reshape(self.value(x), [B, -1, C])        # shape [B, H*W, C]

        # Compute attention scores:
        # tf.matmul(q, k, transpose_b=True) multiplies q with k^T producing [B, H*W, H*W].
        # tf.nn.softmax normalizes scores across the last axis to convert to probabilities.
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True), axis=-1)

        # Multiply attention weights by values: out = attn * v  -> shape [B, H*W, C]
        out = tf.matmul(attn, v)
        # Reshape back to image spatial layout [B, H, W, C]
        out = tf.reshape(out, [B, H, W, C])

        # If user wants the attention map, return it along with the output.
        # Otherwise return the attention-weighted output plus the original input (residual).
        if return_attention:
            # gamma * out scales the attention contribution; adding x forms a residual connection.
            return self.gamma * out + x, attn

        return self.gamma * out + x

# ===============================
# GENERATOR
# ===============================
# The generator maps a random vector (`z`) to an image using upsampling layers.
def build_generator():
    # Input is a latent vector of length LATENT_DIM.
    z = tf.keras.Input(shape=(LATENT_DIM,))

    # Dense: fully connected layer that produces 8*8*256 elements from z.
    # Then Reshape converts it to a small spatial feature map (8x8 with 256 channels).
    x = tf.keras.layers.Dense(8 * 8 * 256)(z)
    x = tf.keras.layers.Reshape((8, 8, 256))(x)

    # BatchNormalization stabilizes and speeds up training by normalizing activations.
    x = tf.keras.layers.BatchNormalization()(x)
    # ReLU introduces non-linearity; used here as activation after normalization.
    x = tf.keras.layers.ReLU()(x)

    # Conv2DTranspose upsamples the spatial size. Arguments: filters, kernel_size, strides, padding.
    # This layer increases (8x8) -> (16x16) by using stride=2.
    x = tf.keras.layers.Conv2DTranspose(128, 4, 2, "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Insert self-attention to allow the generator to model long-range dependencies
    # inside the intermediate feature maps (helps capture global structure).
    x = SelfAttention(128)(x)

    # Further upsample to (32x32)
    x = tf.keras.layers.Conv2DTranspose(64, 4, 2, "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Final upsample to (64x64) and produce 3 color channels.
    # activation="tanh" makes outputs in [-1, 1], matching preprocessing above.
    img = tf.keras.layers.Conv2DTranspose(
        3, 4, 2, "same", activation="tanh"
    )(x)

    # Build and return the Keras Model mapping z -> img.
    return tf.keras.Model(z, img, name="Generator")

# ===============================
# DISCRIMINATOR (WITH SPECTRAL NORMALIZATION)
# ===============================
# The discriminator tries to distinguish real images from fake ones. SpectralNorm is applied
# to convolutional and dense layers to stabilize its training.
def build_discriminator():
    # Input is an image of defined IMG_SIZE and 3 color channels.
    img = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Wrap Conv2D in SpectralNorm to normalize weights before each forward pass.
    # Conv2D arguments: filters, kernel_size, strides, padding
    x = SpectralNorm(
        tf.keras.layers.Conv2D(64, 4, 2, "same")
    )(img)
    # LeakyReLU gives a small slope for negative values which helps gradients flow.
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = SpectralNorm(
        tf.keras.layers.Conv2D(128, 4, 2, "same")
    )(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Self-attention helps the discriminator consider relationships across the whole image.
    x = SelfAttention(128)(x)

    x = SpectralNorm(
        tf.keras.layers.Conv2D(256, 4, 2, "same")
    )(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Flatten prepares features for the final Dense layer that outputs a single score.
    x = tf.keras.layers.Flatten()(x)
    # Final Dense (wrapped with SN) returns a single logit for real/fake decision.
    out = SpectralNorm(tf.keras.layers.Dense(1))(x)

    # Build and return the Keras Model mapping img -> logit.
    return tf.keras.Model(img, out, name="Discriminator")

# ===============================
# BUILD MODELS
# ===============================
# Create instances of generator and discriminator from the builder functions.
generator = build_generator()
discriminator = build_discriminator()

# Optimizers for generator and discriminator.
# Adam is a popular adaptive optimizer. beta_1 and beta_2 control moving averages of gradients.
g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
d_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)

# ===============================
# TRAIN STEP (HINGE LOSS)
# ===============================
# @tf.function compiles this function to a TensorFlow graph for speed.
@tf.function
def train_step(real_images):
    # B stores the actual batch size (may be smaller for the last batch).
    B = tf.shape(real_images)[0]

    # ---- Discriminator update ----
    # Sample random noise for generator input. tf.random.normal samples from a normal distribution.
    noise = tf.random.normal([B, LATENT_DIM])

    # Use GradientTape to record operations for automatic differentiation.
    with tf.GradientTape() as tape:
        # Generate fake images given noise (training=True enables layers like BN to behave in training mode).
        fake_images = generator(noise, training=True)
        # Run discriminator on real images and fake images to get logits (unnormalized scores).
        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        # Hinge loss for discriminator:
        # - tf.nn.relu(1.0 - real_logits) encourages real_logits > 1
        # - tf.nn.relu(1.0 + fake_logits) encourages fake_logits < -1
        # tf.reduce_mean averages values over the batch.
        d_loss = (
            tf.reduce_mean(tf.nn.relu(1.0 - real_logits)) +
            tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
        )

    # Compute gradients of d_loss w.r.t discriminator variables.
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    # Apply gradients using the discriminator optimizer. zip pairs gradients with variables.
    d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))

    # ---- Generator update ----
    # Sample new noise for generator update (could reuse the same noise, but fresh noise is fine).
    noise = tf.random.normal([B, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        # Discriminator's scores for the newly generated images.
        fake_logits = discriminator(fake_images, training=True)
        # Generator loss: tries to maximize discriminator's output for fake images.
        # We use -mean(fake_logits) so minimizing this pushes fake_logits higher.
        g_loss = -tf.reduce_mean(fake_logits)

    # Compute gradients and apply updates to generator variables.
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))

    # Return the scalar losses for logging.
    return d_loss, g_loss

# ===============================
# TRAIN LOOP
# ===============================
# Iterate over epochs and dataset batches, calling train_step on each batch.
for epoch in range(EPOCHS):
    for real_images in dataset:
        # Each real_images is a batch tensor shaped [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        d_loss, g_loss = train_step(real_images)

    # Print losses at the end of each epoch; .numpy() converts tensor to numpy scalar for printing.
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"D Loss: {d_loss.numpy():.4f} | "
        f"G Loss: {g_loss.numpy():.4f}"
    )

# End of file.