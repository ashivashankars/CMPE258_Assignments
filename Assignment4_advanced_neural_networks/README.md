# 🧠 Deep Learning: Data Augmentation, Generalization & Advanced Keras/PyTorch

**Course Assignment — Multi-Colab Series**  
*Covers Part 1 (Regularization & Augmentation) and Part 2 (Advanced Keras/PyTorch Constructs)*

---

## 📁 Repository Structure

```
dl-assignment/
│
├── README.md                          ← You are here
│
├── part1/
│   ├── regularization/
│   │   ├── 01_regularization_tensorflow.ipynb   ← L1/L2, Dropout, Early Stop (TF)
│   │   ├── 02_regularization_pytorch.ipynb      ← Same techniques in PyTorch
│   │   ├── 03_montecarlo_dropout.ipynb          ← MC Dropout for uncertainty
│   │   ├── 04_weight_initializations.ipynb      ← He, Glorot, LeCun, Orthogonal
│   │   ├── 05_batch_normalization.ipynb         ← BatchNorm vs LayerNorm
│   │   ├── 06_custom_dropout_regularizer.ipynb  ← Writing your own from scratch
│   │   └── 07_callbacks_tensorboard_tuner.ipynb ← Callbacks, TensorBoard, KerasTuner
│   │
│   ├── data_augmentation/
│   │   └── 08_keras_cv_augmentation.ipynb       ← KerasCV pipelines & preview
│   │
│   └── modality_augmentation/
│       ├── 09_image_augmentation.ipynb          ← Image aug + classification
│       ├── 10_video_augmentation.ipynb          ← Video frame-level aug
│       ├── 11_text_nlp_augmentation.ipynb       ← nlpaug: synonym, back-translate
│       ├── 12_timeseries_augmentation.ipynb     ← Jitter, scaling, window slicing
│       ├── 13_tabular_augmentation.ipynb        ← SMOTE, Gaussian noise, mixup
│       ├── 14_speech_augmentation.ipynb         ← SpecAugment, pitch shift
│       └── 15_document_image_augmentation.ipynb ← AugLy: doc distortion, stamps
│
└── part2/
    ├── advanced_keras/
    │   ├── 16_custom_lr_scheduler.ipynb         ← OneCycle, WarmupCosine
    │   ├── 17_custom_dropout_normalization.ipynb ← MCAlphaDropout, MaxNormDense
    │   ├── 18_custom_loss_metric.ipynb          ← HuberLoss, HuberMetric
    │   ├── 19_custom_activations_etc.ipynb      ← Activation, initializer, regularizer
    │   ├── 20_custom_layers.ipynb               ← Exponential, MyDense, GaussianNoise
    │   └── 21_custom_model_residual.ipynb       ← ResidualBlock, ResidualRegressor
    │
    └── custom_training/
        ├── 22_custom_optimizer.ipynb            ← MyMomentumOptimizer from scratch
        ├── 23_custom_training_loop.ipynb        ← Fashion-MNIST manual GradientTape
        └── 24_weights_and_biases.ipynb          ← W&B experiment tracking
```

---

## 🎥 Video Walkthrough

> **Full line-by-line video explanation** of every notebook is available at:  
> `[Link to uploaded video — see submission]`

Each section of the video corresponds to numbered notebooks above. Timestamps are in the video description.

---

## 📘 Part 1 — Regularization, Generalization & Data Augmentation

### 🔬 Section A: Regularization Techniques (Notebooks 01–07)

These notebooks implement and compare regularization strategies with **A/B tests** — meaning each technique is trained with and without the technique to directly compare validation curves.

| Notebook | Topic | Framework |
|---|---|---|
| `01_regularization_tensorflow.ipynb` | L1, L2, Dropout, Early Stopping | TensorFlow / Keras |
| `02_regularization_pytorch.ipynb` | Same techniques mirrored | PyTorch |
| `03_montecarlo_dropout.ipynb` | MC Dropout for uncertainty estimation | TF + PyTorch |
| `04_weight_initializations.ipynb` | He, Glorot, LeCun, Orthogonal, zeros | TF + PyTorch |
| `05_batch_normalization.ipynb` | BatchNorm, LayerNorm, GroupNorm | TF |
| `06_custom_dropout_regularizer.ipynb` | Writing custom Dropout & L1 from scratch | TF |
| `07_callbacks_tensorboard_tuner.ipynb` | ModelCheckpoint, EarlyStopping, TensorBoard, KerasTuner | TF |

#### What Each Technique Demonstrates

**L1 / L2 Regularization**
- L1 (Lasso): encourages sparsity — drives small weights to zero
- L2 (Ridge): penalizes large weights — keeps weights small but non-zero
- Both are added directly to `kernel_regularizer` in Keras layers
- A/B test shows training vs. validation loss with and without regularization

**Dropout**
- Randomly zeros activations during training — forces redundant representations
- Rate typically between 0.2–0.5; higher = more regularization
- Disabled at inference time automatically in Keras/PyTorch

**Early Stopping**
- Monitors validation loss; stops when it stops improving for `patience` epochs
- `restore_best_weights=True` ensures model reverts to best checkpoint

**Monte Carlo Dropout**
- Keep dropout active at inference time and run N forward passes
- Mean = prediction, Std = uncertainty estimate
- Useful for Bayesian deep learning and safety-critical applications

**Weight Initializations**
- `glorot_uniform` (Xavier): default for tanh/sigmoid activations
- `he_normal` (Kaiming): best for ReLU-family activations
- `lecun_normal`: designed for SELU activations
- `orthogonal`: good for RNNs — preserves gradient magnitude
- Rule of thumb: match initializer to activation function

**Batch Normalization**
- Normalizes layer inputs per mini-batch during training
- Allows higher learning rates, reduces sensitivity to initialization
- Place before or after activation (pre-activation BN often better for deep nets)

**Custom Dropout & Custom Regularizer**
- Subclass `tf.keras.layers.Layer` to write your own dropout variant
- Subclass `tf.keras.regularizers.Regularizer` for custom penalty terms

**Callbacks & TensorBoard**
- `ModelCheckpoint`: saves model at best validation epoch
- `ReduceLROnPlateau`: halves LR when validation plateaus
- `TensorBoard`: visualize loss curves, histograms, embeddings live
- Launch with `%tensorboard --logdir logs/`

**KerasTuner**
- Automatically searches hyperparameter space (LR, units, dropout rate)
- Supports `RandomSearch`, `BayesianOptimization`, `Hyperband`

---

### 🖼️ Section J: KerasCV Data Augmentation (Notebook 08)

Uses the modern `keras_cv` library for GPU-accelerated augmentation pipelines:
- `RandomFlip`, `RandomRotation`, `RandomCrop`, `RandAugment`
- `CutMix`, `MixUp` — label-mixing augmentation strategies
- Preview augmented batches visually before training
- End-to-end image classification pipeline with augmentation baked in

---

### 🗂️ Section K: Multi-Modality Augmentation (Notebooks 09–15)

Each notebook follows the same structure:
1. Load a real dataset
2. Apply augmentation techniques
3. Visualize/listen to augmented samples
4. Train a classifier — compare augmented vs. baseline

| Modality | Library | Key Techniques |
|---|---|---|
| **Image** | `tf.keras.layers`, `albumentations` | Flip, Crop, ColorJitter, Cutout, MixUp |
| **Video** | `pytorchvideo`, OpenCV | Frame drop, temporal jitter, spatial crop |
| **Text / NLP** | `nlpaug` | Synonym replace, back-translation, insertion |
| **Time Series** | `tsaug` | Jitter, scaling, window slicing, drift |
| **Tabular** | `imblearn` (SMOTE), custom | Gaussian noise, feature dropout, Mixup |
| **Speech** | `audiomentations` | Pitch shift, time stretch, SpecAugment |
| **Document Images** | `AugLy` (Facebook) | Distortion, stamps, low quality, overlays |

---

## 📗 Part 2 — Advanced Keras & Custom Training

All notebooks reference and extend examples from Aurélien Géron's *Hands-On ML* chapters 11–13.

### 🔧 Advanced Constructs (Notebooks 16–24)

| Notebook | Custom Component | Key Class / Pattern |
|---|---|---|
| `16_custom_lr_scheduler.ipynb` | Learning Rate Scheduler | `OneCycleScheduler`, `WarmupCosineDecay` |
| `17_custom_dropout_normalization.ipynb` | Dropout + Normalization | `MCAlphaDropout`, `MaxNormDense` |
| `18_custom_loss_metric.ipynb` | Loss + Metric | `HuberLoss`, `HuberMetric` |
| `19_custom_activations_etc.ipynb` | Activation, Initializer, Regularizer | `leaky_relu`, `MyGlorotInitializer`, `MyL1Regularizer` |
| `20_custom_layers.ipynb` | Layers | `ExponentialLayer`, `MyDense`, `AddGaussianNoise` |
| `21_custom_model_residual.ipynb` | Full Model | `ResidualBlock`, `ResidualRegressor` |
| `22_custom_optimizer.ipynb` | Optimizer | `MyMomentumOptimizer` |
| `23_custom_training_loop.ipynb` | Training Loop | `GradientTape`, Fashion-MNIST |
| `24_weights_and_biases.ipynb` | Experiment Tracking | `wandb.init`, sweeps, artifact logging |

#### Key Concepts Explained

**Custom Learning Rate Scheduler**
- Subclass `tf.keras.callbacks.Callback`, override `on_epoch_begin`
- OneCycle: LR ramps up then decays — enables fast convergence
- Use `LearningRateScheduler` callback or `tf.keras.optimizers.schedules`

**Custom Dropout (MCAlphaDropout)**
- Extends `AlphaDropout` to stay active during inference
- Enables Bayesian uncertainty quantification at prediction time

**Custom Normalization (MaxNormDense)**
- Dense layer with max-norm constraint on kernel weights
- Prevents any single weight from growing too large
- Implemented via `tf.keras.constraints.MaxNorm`

**Custom Loss (HuberLoss)**
- Behaves like MSE for small errors, MAE for large — robust to outliers
- Subclass `tf.keras.losses.Loss`, implement `call(y_true, y_pred)`
- `threshold` is a configurable hyperparameter saved via `get_config()`

**Custom Metric (HuberMetric)**
- Subclass `tf.keras.metrics.Metric`
- Must implement `update_state`, `result`, and `reset_state`

**Custom Activation / Initializer / Regularizer / Constraint**
- Functions or subclassed objects passed directly into layer constructors
- Must be serializable (implement `get_config`) for model saving

**Custom Layers**
- Subclass `tf.keras.layers.Layer`
- Implement `build(input_shape)` to create weights
- Implement `call(inputs)` for the forward pass
- `training` argument controls behavior differences (e.g., dropout)

**Custom Model (Residual)**
- Subclass `tf.keras.Model`, define skip connections in `call()`
- `ResidualBlock`: adds input to output (skip connection)
- `ResidualRegressor`: stacks residual blocks for regression

**Custom Optimizer**
- Subclass `tf.keras.optimizers.Optimizer`
- Implement `_resource_apply_dense` with the update rule

**Custom Training Loop**
- Use `tf.GradientTape()` context manager
- Manually call `tape.gradient`, `optimizer.apply_gradients`
- Full control: multi-output losses, gradient clipping, logging

**Weights & Biases (W&B)**
- `wandb.init(project=...)` starts a run
- `wandb.log({"loss": ...})` logs metrics per step
- Sweeps: define a YAML config, W&B agent tries combinations
- Artifacts: version datasets and model checkpoints

---

## ⚙️ Setup & Dependencies

### Google Colab (Recommended)
All notebooks are designed to run on Google Colab with a free GPU. Click the badge at the top of each notebook.

```python
# Run this at the top of any notebook to install dependencies
!pip install tensorflow keras-cv torch torchvision nlpaug tsaug \
             audiomentations albumentations augly imbalanced-learn \
             keras-tuner wandb -q
```

### Local Setup
```bash
git clone https://github.com/<your-username>/dl-assignment.git
cd dl-assignment
pip install -r requirements.txt
jupyter lab
```

---

## 📚 References & Resources

| Resource | Link |
|---|---|
| Hands-On ML3 (Géron) — Chapter 10 | [github.com/ageron/handson-ml3 — ch10](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb) |
| Hands-On ML3 — Chapter 11 | [github.com/ageron/handson-ml3 — ch11](https://github.com/ageron/handson-ml3/blob/main/11_training_deep_neural_networks.ipynb) |
| Hands-On ML2 — Chapter 11 | [github.com/ageron/handson-ml2 — ch11](https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb) |
| Hands-On ML2 — Chapter 12 | [github.com/ageron/handson-ml2 — ch12](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb) |
| Hands-On MLP (PyTorch) | [github.com/ageron/handson-mlp](https://github.com/ageron/handson-mlp) |
| TensorFlow Augmentation Tutorial | [tensorflow.org/tutorials/images/data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation) |
| KerasCV Docs | [keras.io/keras_cv](https://keras.io/keras_cv) |
| AugLy (Facebook Research) | [github.com/facebookresearch/AugLy](https://github.com/facebookresearch/AugLy) |
| Data Augmentation Review | [github.com/AgaMiko/data-augmentation-review](https://github.com/AgaMiko/data-augmentation-review) |
| Awesome Data Augmentation | [brunokrinski.github.io/awesome-data-augmentation](https://brunokrinski.github.io/awesome-data-augmentation/) |

---

## 📝 Notes on A/B Testing Methodology

Throughout Part 1, each regularization notebook follows this structure:

1. **Baseline model** — no regularization, trained to overfitting
2. **Regularized model** — same architecture with technique applied
3. **Side-by-side plots** — training vs. validation loss/accuracy for both
4. **Conclusions** — when to use each technique and why

The goal is not just to show the technique works, but to understand *when* and *why* to reach for it.

