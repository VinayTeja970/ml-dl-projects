
# Reward-Guided Text Generation for Agriculture using GPT-2

## Project Overview

This project implements a reward-guided text generation system using GPT-2, optimized for agriculture-related prompts. The training loop is designed to simulate reinforcement learning using custom reward signals from multiple transformer models. The goal is to generate helpful, relevant responses to agricultural queries by guiding the language model based on sentiment and helpfulness scores.

---

## Technical Summary

### Models and Tools
- **Language Model**: GPT2LMHeadModel
- **Reward Models**:
  - `distilbert-base-uncased-finetuned-sst-2-english` (sentiment classification)
  - `facebook/bart-large-mnli` (zero-shot helpfulness classification)
- **Additional Tools**: `transformers`, `datasets`, `torch`, `matplotlib`

---

## Reward Logic

Custom reward function combines:

1. **BERT Sentiment Score**
   - Rewards higher scores for "POSITIVE" sentiment.

2. **Zero-Shot Helpfulness Score**
   - Classifies generated text as "helpful" vs "not helpful".

3. **Keyword Match Bonus**
   - Keywords like `"fertilizer"`, `"irrigation"`, `"crop"` boost the score.

Final reward:
```python
reward = (sentiment_score + helpfulness_score) / 2.0 + keyword_bonus
```

---

## Dataset

- **Source**: Hugging Face Dataset – `Mahesh2841/Agriculture`
- **Fields**: `instruction`, `input`, `response`
- **Usage**: Only the `input` field is used as the prompt.

---

## Training Loop

- **Epochs**: 5
- **Steps**:
  - Random sample prompt is selected.
  - GPT-2 generates text.
  - Text is scored using the reward function.
  - Negative reward * log-probabilities used as the loss.
  - Optimizer performs a backpropagation step to improve future generations.

---

## Output Visualization

- A matplotlib plot shows **reward progression over epochs**, helping monitor learning trends.

---

## Strengths

- Domain-specific application using a general-purpose language model.
- Reward shaping using both classification and keyword heuristics.
- Modular design allows future extension to other domains or more refined reward functions.

---

## Potential Improvements

- Use **AgriBERT** or fine-tuned LLMs for more relevant evaluation.
- Support multi-turn question answering.
- Extend reward function with factuality checks or human feedback.
- Integrate with chatbot interfaces or deploy via web for real-time use.

---

## Author

**Vinay Teja Pathakamuri**  
Purdue University Northwest  

