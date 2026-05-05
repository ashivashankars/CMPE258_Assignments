# 🧠 Deep Learning & NLP Portfolio — Google Colab Notebooks

A curated portfolio of executed deep learning notebooks covering RNNs/LSTMs, NLP, Vision Transformers, and Graph Transformers. Each notebook is accompanied by a walkthrough video explaining the code block by block, including all inputs and outputs.

---

## 📺 Video Walkthroughs

| # | Topic | Colab Notebook | YouTube Walkthrough |
|---|-------|---------------|---------------------|
| 1 | RNN & LSTM | [Open Colab](https://colab.research.google.com/drive/1uGPf1pUz_P0JIlpBqcPOJyHikmvO8uV2) | [▶ Video](#) *(recording in progress)* |
| 2 | NLP Fundamentals | [Open Colab](https://colab.research.google.com/drive/129nxLYowmTdGpQGziABOYs7FpRqQmNsF) | [▶ Video](#) *(recording in progress)* |
| 3 | Vision Transformers (ViT) | [Open Colab](https://colab.research.google.com/drive/1IQp0RU4w7DXRKlLITyYgYFLMntFvrvdx) | [▶ Video](#) *(recording in progress)* |
| 4 | Graph Transformers | [Open Colab](https://colab.research.google.com/drive/1mOImVS1KcjpIFESouEemLeiN-Y-l4Whj) | [▶ Video](#) *(recording in progress)* |

> 📌 **Note:** Replace the `#` placeholders above with actual YouTube video URLs after recording and uploading each walkthrough.

---

## 📁 Repository Structure

```
├── README.md
├── notebooks/
│   ├── 01_rnn_lstm/
│   │   ├── rnn_lstm_executed.ipynb       # Colab copy with all outputs
│   ├── 02_nlp/
│   │   ├── nlp_executed.ipynb            # Colab copy with all outputs
│   ├── 03_vision_transformers/
│   │   ├── vit_executed.ipynb            # Colab copy with all outputs
│   └── 04_graph_transformers/
│       ├── graph_transformer_executed.ipynb   # Colab copy with all outputs
```

---

## 📓 Notebook Summaries

### 1. 🔁 RNN & LSTM
**Colab:** https://colab.research.google.com/drive/1uGPf1pUz_P0JIlpBqcPOJyHikmvO8uV2

**What it covers:**
- Recurrent Neural Network (RNN) fundamentals — how hidden states carry sequence context
- The vanishing gradient problem and why LSTMs were invented
- LSTM cell architecture — forget gate, input gate, output gate, and cell state
- Building and training an LSTM model in PyTorch/Keras on sequential data
- Visualizing training loss and evaluating model performance

**Key concepts demonstrated:**
- Sequence modeling and time series prediction
- Backpropagation Through Time (BPTT)
- Gated memory cells and long-range dependencies

**Video breakdown:**
Each code block in the video covers: data preparation → model architecture → training loop → loss visualization → predictions vs actuals.

---

### 2. 📝 NLP Fundamentals
**Colab:** https://colab.research.google.com/drive/129nxLYowmTdGpQGziABOYs7FpRqQmNsF

**What it covers:**
- Text preprocessing: tokenization, stopword removal, stemming/lemmatization
- Word embeddings: Word2Vec, GloVe representations
- Bag of Words and TF-IDF vectorization
- Sentiment analysis / text classification pipeline
- Using pre-trained transformer models (BERT/HuggingFace) for NLP tasks

**Key concepts demonstrated:**
- Converting raw text to numerical representations
- Semantic similarity and vector space models
- Fine-tuning vs. feature extraction with pre-trained models

**Video breakdown:**
Each code block covers: raw text ingestion → preprocessing pipeline → feature engineering → model training → evaluation metrics (accuracy, F1).

---

### 3. 👁️ Vision Transformers (ViT)
**Colab:** https://colab.research.google.com/drive/1IQp0RU4w7DXRKlLITyYgYFLMntFvrvdx

**What it covers:**
- The original Vision Transformer (ViT) paper architecture: "An Image is Worth 16×16 Words"
- Patch embedding — splitting images into fixed-size patches treated as tokens
- Positional encoding for 2D image patches
- Multi-head self-attention applied to image data
- Training ViT on image classification benchmarks (CIFAR-10, ImageNet subsets)
- Attention map visualization — what the model "looks at"

**Key concepts demonstrated:**
- Applying transformer architecture to computer vision
- Inductive bias comparison: CNNs vs. ViTs
- Patch tokenization and the [CLS] token for classification

**Video breakdown:**
Each code block covers: image patching → embedding projection → transformer encoder blocks → classification head → attention visualization.

---

### 4. 🕸️ Graph Transformers
**Colab:** https://colab.research.google.com/drive/1mOImVS1KcjpIFESouEemLeiN-Y-l4Whj

**What it covers:**
- Graph Neural Network (GNN) fundamentals — nodes, edges, message passing
- Limitations of standard GNNs: over-smoothing, limited long-range propagation
- Graph Transformer architecture — combining attention with graph structure
- Node classification and graph classification tasks
- Benchmarks on molecular/social graph datasets

**Key concepts demonstrated:**
- Message passing neural networks (MPNN)
- Incorporating graph topology into attention mechanisms
- Spectral vs. spatial graph convolutions
- Positional encodings for graphs (Laplacian eigenvectors)

**Video breakdown:**
Each code block covers: graph data loading (PyTorch Geometric) → adjacency/feature setup → transformer layer → pooling → classification output.

---

## 🛠️ Setup & Requirements

All notebooks are designed to run on **Google Colab** (free tier is sufficient for most). For local execution:

```bash
pip install torch torchvision transformers datasets
pip install torch-geometric  # for Graph Transformers
pip install matplotlib seaborn scikit-learn
```

**Recommended runtime:** GPU (T4 or higher) for Vision Transformers and Graph Transformers.

---

## 📚 Learning Resources

| Topic | Recommended Reading |
|-------|-------------------|
| RNN/LSTM | [Understanding LSTMs – Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| NLP & Transformers | [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/) |
| Vision Transformers | [ViT Paper – Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929) |
| Graph Transformers | [A Generalization of Transformers to Graphs](https://arxiv.org/abs/2012.09699) |

---

## 👤 About

This repository is part of a Deep learning portfolio. All notebooks have been executed with real outputs preserved, and video walkthroughs are recorded to reinforce learning and showcase understanding of each architecture.

**To follow along:** Clone this repo, open any `.ipynb` in Colab or Jupyter, and refer to the corresponding video for a guided explanation.

---

*Last updated: May 2026*
