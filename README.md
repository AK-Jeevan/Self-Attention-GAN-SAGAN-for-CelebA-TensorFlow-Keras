# 👁️ Self-Attention Generative Adversarial Network (SAGAN)  
**TensorFlow / Keras**

This repository demonstrates a complete **Self-Attention Generative Adversarial Network (SAGAN)** implementation using TensorFlow and Keras, trained on the CelebA dataset to generate realistic human face images.

SAGAN introduces **self-attention mechanisms** to GANs, enabling the generator and discriminator to capture **long-range spatial dependencies**. Combined with **spectral normalization**, SAGAN achieves exceptional training stability and image quality by allowing every spatial position to attend to all other positions in the feature maps.

---

## 🚀 What This Project Covers

- Loading and preprocessing **CelebA face dataset** to `[-1, 1]` range
- Building a **generator with self-attention** for high-quality face synthesis
- Building a **discriminator with spectral normalization** for stable training
- Implementing **self-attention layers** to capture global spatial dependencies
- Implementing **spectral normalization** via power iteration algorithm
- Learnable **gamma parameter** for safe attention contribution control
- **Hinge loss** formulation for stable adversarial training
- Custom training loop with `tf.GradientTape` and optimal gradient flow
- Efficient dataset pipeline with `tf.data` API
- Multiple upsampling and downsampling blocks with attention

---

## 🧠 Why Use SAGAN?

This project helps you:

- Understand **self-attention mechanisms in generative models**
- Learn about **spectral normalization** for discriminator stability
- Master **long-range spatial dependency modeling**
- Generate **high-quality, diverse face images**
- Implement **production-ready attention-based GANs**
- Apply **state-of-the-art generative modeling techniques**
- Understand **gamma-scaled residual connections** for attention

SAGAN is especially beneficial when:
- You need to capture **global context** in generated images
- Local convolutional receptive fields are **insufficient**
- Training stability is critical
- Generating **high-fidelity faces or complex scenes** is required
- You want models that can attend to **multiple regions simultaneously**

---

## 🏗️ Training Architecture

### 🔹 Generator
- **Input**: Random latent vector (`latent_dim = 128`)
- **Architecture**:
  - Dense → Reshape to 8 × 8 × 256 feature map
  - Batch Normalization + ReLU
  - Conv2DTranspose: 8 × 8 → 16 × 16 (128 filters)
  - **Self-Attention** layer (16 × 16 scale)
  - Conv2DTranspose: 16 × 16 → 32 × 32 (64 filters)
  - Conv2DTranspose: 32 × 32 → 64 × 64 (3 channels)
- **Output**: 64 × 64 × 3 RGB face images with **tanh activation** ([-1, 1] range)
- **Purpose**: Maps random noise to realistic CelebA face images
- **Self-Attention**: Enables generator to coordinate features across entire image

### 🔹 Discriminator (with Spectral Normalization)
- **Input**: Image (64 × 64 × 3)
- **Architecture**:
  - SpectralNorm + Conv2D: 64 × 64 → 32 × 32 (64 filters) + LeakyReLU(0.2)
  - SpectralNorm + Conv2D: 32 × 32 → 16 × 16 (128 filters) + LeakyReLU(0.2)
  - **Self-Attention** layer (16 × 16 scale)
  - SpectralNorm + Conv2D: 16 × 16 → 8 × 8 (256 filters) + LeakyReLU(0.2)
  - Flatten → SpectralNorm + Dense(1)
- **Output**: Single logit score for real/fake classification
- **Purpose**: Distinguishes real faces from generated ones
- **Spectral Normalization**: Constrains largest singular value of weight matrices
- **Self-Attention**: Allows discriminator to verify global image consistency

### 🔹 Self-Attention Layer (SAGAN)
- **Input**: Feature map with shape [B, H, W, C]
- **Mechanism**:
  - Query: 1×1 Conv mapping to C/8 dimensions
  - Key: 1×1 Conv mapping to C/8 dimensions
  - Value: 1×1 Conv mapping to C dimensions
  - Attention: Softmax(Q·K^T) ∈ [0, 1]
  - Output: γ·(Attention·Value) + x (residual connection)
- **Gamma**: Learnable scalar initialized to 0, enables gradual attention learning
- **Complexity**: O(HW × HW) but applied at intermediate feature scales

### 🔹 Spectral Normalization
- Constrains **largest singular value** of weight matrices to 1
- Uses **power iteration** to approximate top singular vector
- Applied to Conv2D and Dense layers in discriminator
- Stabilizes discriminator training without explicit loss penalty
- Standard in modern GANs (BigGAN, SAGAN, StyleGAN)

---

## 🧪 Training Strategy

- **Loss Function**: Hinge loss (non-saturating, stable)
- **Discriminator Loss**: Encourages real logits > 1, fake logits < -1
  - Provides stronger gradients compared to BCE
- **Generator Loss**: Maximizes discriminator scores on generated images
- **Optimizers**: Adam with `beta_1 = 0.0` (recommended for GANs)
  - Learning rate: `1e-4` for both networks
- **Data Pipeline**: tf.data with AUTOTUNE prefetching
- **Batch Normalization**: Stabilizes generator training
- **LeakyReLU**: Discriminator activation (0.2 slope)

---

## 📉 Loss Functions

### Discriminator Loss (Hinge)
```
L_disc = E[ReLU(1 - D(real))] + E[ReLU(1 + D(fake))]
```
- Encourages D(real) > 1 for real images
- Encourages D(fake) < -1 for fake images
- Provides stronger gradients throughout training

### Generator Loss
```
L_gen = -E[D(fake)]
```
- Minimizes negative discriminator score on fake images
- Encourages generator to produce images discriminator considers real
- Non-saturating loss prevents gradient vanishing

---

## 🔍 Key Concepts Demonstrated

- **Self-Attention (SAGAN)**: Computing attention between all spatial positions
- **Spectral Normalization**: Constraining discriminator Lipschitz constant
- **Power Iteration**: Approximating largest singular value efficiently
- **Gamma-scaled Attention**: Learned blending of attention and residual paths
- **Hinge Loss**: Non-saturating loss for stable GAN training
- **Efficient Dataset Pipeline**: tf.data.Dataset with AUTOTUNE and prefetch
- **Custom Training Loop**: Fine-grained control with tf.GradientTape
- **Convolutional Upsampling/Downsampling**: Spatial dimension manipulation
- **Batch Normalization in Generator**: Training stability
- **LeakyReLU in Discriminator**: Gradient flow preservation

---

## 💾 Output Artifacts

After training, the script generates:

- **Generator model** (`generator`) for face synthesis
- **Discriminator model** (`discriminator`) for classification
- **Training logs** printed each epoch with D and G losses

Generated faces can be sampled by:
```python
noise = tf.random.normal([16, LATENT_DIM])
faces = generator(noise, training=False)
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMG_SIZE` | 64 | CelebA image resolution (square) |
| `BATCH_SIZE` | 64 | Training batch size |
| `LATENT_DIM` | 128 | Noise vector dimensionality |
| `EPOCHS` | 5 | Total training epochs |
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| `beta_1` | 0.0 | Adam momentum (GAN-optimized) |
| `beta_2` | 0.9 | Adam exponential moving average |
| LeakyReLU slope | 0.2 | Discriminator activation slope |
| Power iterations | 1 | Spectral norm approximation steps |

---

## 📊 Self-Attention Details

### Why Attention in GANs?

Convolutional layers have **limited receptive fields**. Self-attention allows:
- Each pixel to "see" the entire feature map
- Long-range dependencies modeling
- Global image structure consistency
- Better coherence in generated images

### Gamma Parameter

- Initialized to **0** (attention has no effect at start)
- Learned during training (becomes positive)
- Allows safe integration without disrupting pre-training
- Prevents attention from dominating gradients early

### Attention Complexity

- Standard attention: O(HW)² where HW is spatial dimension
- Applied at intermediate scales (H=W=16) to reduce cost
- Spatial softmax ensures values sum to 1

---

## 🔬 Spectral Normalization Details

### Power Iteration Algorithm

1. Initialize random vector **u** of shape [1, C]
2. Compute **v** ← normalize(u·W^T)
3. Compute **u** ← normalize(v·W)
4. σ ≈ v·W·u^T (largest singular value)
5. Normalize: W_norm = W / σ

### Benefits

- **1-Lipschitz continuity**: Discriminator gradients remain bounded
- **Prevents spectral collapse**: Eigenvalues don't explode
- **Simple to implement**: Just one power iteration per forward pass
- **Minimal computational cost**: Single matrix multiplication

---

## ⚠️ Important Notes

- **Gamma Initialization**: Critical that gamma starts at 0 for stability
- **Spectral Norm**: Only applied to discriminator, not generator
- **Attention Placement**: Typically at intermediate feature scales (64 filters)
- **No BatchNorm in Discriminator**: Standard practice to avoid statistics mismatch
- **CelebA Dataset**: Automatically downloaded via tensorflow_datasets
- **Hinge Loss**: Different from BCE; requires careful tuning of margin (1.0)

---

## 📊 Expected Training Behavior

- **Epoch 1-2**: Losses stabilize; generator learns basic face structure
- **Epoch 2-4**: Discriminator loss plateaus; generator loss decreases
- **Epoch 4-5**: Generated faces become sharper and more diverse
- **Attention Maps**: Initially uniform; becomes concentrated on facial features
- **Gamma Values**: Gradually increase from 0 toward 0.1-0.3

---

## 🎯 Advanced Modifications

Potential extensions to this implementation:

- **Progressive growing**: Start at low resolution, gradually increase
- **Class conditioning**: Add class labels for conditional face generation
- **Perceptual losses**: Use pretrained VGG19 for better feature matching
- **Multi-scale discriminator**: Process images at multiple resolutions
- **Attention visualization**: Visualize which regions the model attends to
- **Gradient penalty**: Add WGAN-GP style regularization

---

## 📜 License

MIT License

---

## ⭐ Support

If this repository helped you:

⭐ Star the repo  
🧠 Share it with other GAN and deep learning learners  
🚀 Use it as a foundation for advanced face generation projects  
📖 Reference it in research or technical articles about attention mechanisms
