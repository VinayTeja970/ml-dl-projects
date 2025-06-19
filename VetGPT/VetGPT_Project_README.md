
# VetGPT: Veterinary Language Model Prototype

## Project Overview

VetGPT is a domain-specific language model fine-tuned for veterinary science use cases. It leverages a mix of web-scraped data, textbook content, and transformer-based training to support the generation and retrieval of veterinary-related content.

---

## Project Files and Data Sources

### 1. Web Scraping Notebook (`VetGPT_Data_Scraping.ipynb`)
- Purpose: Scrape articles from [MSD Veterinary Manual](https://www.msdvetmanual.com/) using Scrapy.
- Steps:
  - Installs required Python packages (`scrapy`, `torch`, etc.)
  - Defines a `Scrapy` spider `MsdSpider` that extracts article titles and content.
  - Follows pagination via `a.next` links.
  - Exports data to a CSV file (`msd_data.csv`) for further processing or training.

### 2. Training Notebook (`VetGPT_Limited Data.ipynb`)
- Purpose: Train a GPT-style character-level language model from scratch using PyTorch.
- Training Dataset: `corrected_vet_data1.txt`, a text file composed of cleaned veterinary content.

#### Key Stages in the Notebook:

##### Environment Setup
- Libraries: `torch`, `numpy`, `torch.nn`, etc.
- Device: Automatically uses GPU if available (`cuda`), else CPU.

##### Hyperparameters
```python
block_size = 40
batch_size = 64
vocab_size = 88
n_embd = 512
n_head = 8
n_layer = 6
learning_rate = 0.00035
max_iters = 6000
```

##### Data Loading & Preprocessing
- Loads character-level text from `corrected_vet_data1.txt`
- Builds a vocabulary using unique characters.
- Encodes the full text to numerical format.
- Splits into training (90%) and validation (10%).

##### Tokenization
```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
```

##### GPT Architecture (from scratch)
- Components Implemented:
  - `Head`: Self-attention head with masking
  - `MultiHeadAttention`: Multiple parallel heads
  - `FeedForward`: Non-linear transformation
  - `Block`: Transformer block
  - `GPTModel`: Complete stack of layers with embeddings and final projection

##### Training Loop
```python
for iter in range(max_iters):
    if iter % eval_interval == 0:
        print(f"step {iter}: train loss ... val loss ...")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
```

##### Text Generation
- Initial tokens (e.g., `encode("rabies in dogs")`) used to generate veterinary content.
- Uses `generate()` method on trained model.

---

## Supporting Text Files

### `ivri_data.txt`
- Contains educational and administrative information from the ICAR-IVRI veterinary institute.
- Covers:
  - Technology dissemination systems
  - ATIC (Agricultural Technology Information Centres)
  - Postgraduate admission guidelines

### `VeterinaryPG.txt`
- Contains structured content from a veterinary pharmacology book.
- Focuses on:
  - Drug administration
  - Clinical safety
  - Pharmacy law
  - Nursing practices

---

## Tools & Libraries Used

- Python
- Scrapy for web scraping
- Torch for deep learning
- PDFPlumber for PDF parsing (used in earlier preprocessing)
- Transformers for grammar correction & formatting

---

## Observations

- The initial GPT model gives relevant but not always accurate responses.
- With just 6000 iterations and limited data, fine-tuning is incomplete.
- Improved performance expected with:
  - More domain-specific data
  - Extended training iterations
  - Larger context size

---

## Future Improvements

- Use token-level GPT models (e.g., GPT-2, GPT-Neo) instead of character-level
- Expand dataset with:
  - ICD codes
  - Case studies
  - More academic content
- Implement a web-based chatbot interface for real-world vet queries
- Incorporate image-based analysis (e.g., plant or pet disease detection)

---

## Author

**Vinay Teja Chowdary Pathakamuri**  
Purdue University Northwest  

