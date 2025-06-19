
# Streamlined Stable Diffusion

## Project Overview

This project explores the generation of images from text using Stable Diffusion, with a specific focus on Pokémon character image generation.The work showcases practical implementation of generative AI under limited compute constraints.

**Team Members:**  
- Srinivas Subnivis  
- Vinay Teja Pathakamuri  
- Likhita Kolagani

---

## Objectives

1. Generate images from captions using CLIP-based embedding models.
2. Generate Pokémon character images from textual prompts.
3. Address hardware limitations (CUDA memory issues) during model training.

---

## Datasets Used

- [Emotion Recognition Dataset](https://www.kaggle.com/api/v1/datasets/download/sujaykapadnis/emotion-recognition-dataset?dataset_version_number=1)
- [Pokémon Image Dataset (First Generation)](https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files)

---

## Key Challenges and Solutions

### 1. CLIP-Based Captioned Image Generation
- Used a dataset with 32,000 images and 5 captions per image.
- Attempted to use CLIP embeddings with Hugging Face APIs.
- Due to integration issues, the team decided to pivot focus.

### 2. Pokémon Image Generation
- Used a dataset of ~15,000 Pokémon images, multiple views per character.
- Input: Pokémon name as text
- Output: AI-generated image of the Pokémon
- Encountered CUDA out-of-memory issues. Resolved by reducing batch size from 1024 to 2.

---

## Results

Successfully generated images for the following Pokémon:
- Pikachu
- Spearow
- Charmeleon

Though some images were blurry, they closely matched the visual characteristics of the intended Pokémon, validating the effectiveness of the adjusted pipeline.

---

## Technologies Used

- Stable Diffusion
- CLIP (Contrastive Language-Image Pretraining)
- Hugging Face
- PyTorch
- Python
- CUDA

---

## Conclusion

This project highlighted the viability of running diffusion-based generative models on constrained hardware by carefully managing training configurations. It also demonstrated the flexibility of these models when applied to niche domains like character-based image generation.

---

## Author

**Vinay Teja Pathakamuri**  
Purdue University Northwest  

