# Language Modeling: Statistical N-Grams vs Recurrent Neural Networks

## Project Overview

This project involves the development and evaluation of two distinct language modeling approaches: a Statistical n-gram model and a Neural RNN model. Both models were trained and evaluated on the **NLTK Brown corpus (news category)** using a strictly shared data pipeline to ensure a fair comparison.

---

## Data Preparation (Part D)

* **Source**: NLTK Brown corpus, category: 'news'.


* **Preprocessing**:
* Converted all tokens to lowercase.


* Removed punctuation-only tokens while preserving contractions like "don't".


* Added special tokens: `<bos>` (start of sentence), `<eos>` (end of sentence), and `<unk>` (unknown words).




* **Data Split**:
* 80% Train, 10% Validation, 10% Test.


* Splitting was performed by sentence to prevent data leakage.




* **Vocabulary**:
* Built strictly from the training set with a `min_freq` of **3**.


* Tokens appearing fewer than 3 times were mapped to `<unk>`.





---

## Part A: Statistical n-gram Language Model

* **Model Type**: Trigram (n=3) from NLTK library with Laplace (Add-1) smoothing.


* **Smoothing Necessity**: Smoothing is required to prevent the model from assigning zero probability to unseen trigrams, which would otherwise result in infinite perplexity.



### Results

* **Validation Perplexity**: ~2029.09
* **Test Perplexity**: ~1991.85
* **Perplexity Factors**: Vocabulary size (larger vocab increases perplexity), the quality/diversity of the data, the choice of smoothing, and the context window (n-gram length).
* **Observations**: The high perplexity is expected for Laplace smoothing on a small dataset, as it dilutes probability mass across a massive number of unseen combinations.
* **Text Generation**: Generated text showed high local grammatical correctness (3-word windows) but lacked global coherence.



---

## Part B: Neural Language Model (RNN)

* **Architecture**: PyTorch-based RNN featuring an `nn.Embedding` layer, an `nn.RNN` layer (with `batch_first=True`), and an `nn.Linear` output layer.


* **Hardware Acceleration**: Trained on Apple Silicon (M4) using the **MPS (Metal Performance Shaders)** backend.

### Final Hyperparameters

* **Embedding Dimension**: 128 


* **Hidden Dimension**: 256 


* **Number of Layers**: 1 


* **Learning Rate**: 1e-3 


* **Sequence Length**: 30 


* **Total Parameters**: 1,466,336



### Training Performance

* **Observations**: The model exhibited rapid convergence. By Epoch 2, the model began to overfit the training data significantly, leading to a rise in validation perplexity., despite decreasing training loss.
* **Strategy**: Early stopping was utilized, saving the model state after **Epoch 2** to capture the best generalization.

### Results

* **Final Test Perplexity**: ~5625.51, at Epoch 5. (~758.50 at Epoch 2)


* **Comparison**: The RNN achieved significantly higher perplexity than the n-gram model at Epoch 5, but had a lower perplexity at Epochs 1 and 2.


* **Why?**: Unlike the n-gram model, the RNN uses dense embeddings to capture semantic relationships and shares parameters across time steps, allowing it to generalize much more effectively to unseen sequences. Despite this, the RNN overfits because its high parameter capacity relative to the small dataset allows it to memorize specific training sequences rather than learning generalizable linguistic patterns. Without regularization like dropout, the model effectively stores the unique signatures of the training data in its hidden states and embeddings.



### Text Generation Analysis

* **Sample (Epoch 2)**: *"`<bos>` the two climate of central and local governments to <unk> three new members to have their support to the attention of the congolese government for the <unk> of communism throughout southeast asia `<eos>`"*
* **Grammaticality**: High; the model correctly handles basic English syntax.


* **Coherence**: Moderate; while locally logical, the sentence lacks a complex overarching narrative.


* **Repetition**: Low; the model successfully avoided infinite loops in the first 30 tokens.


* **Long-range Dependencies**: The model successfully maintained subject-verb agreement over short-to-medium spans.
