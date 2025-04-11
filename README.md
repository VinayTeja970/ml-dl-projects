**Humor Generation with GPT-2 using Reinforcement Learning from Human Feedback (RLHF)**
This project explores fine-tuning GPT-2 Medium using Reinforcement Learning with Human Feedback (RLHF) to generate humorous text completions. The model is rewarded based on a pretrained BERT humor classifier, which scores the generated outputs on their humor content.

**Project Structure**
**Humor_RLHF_GPT2.ipynb:** The core notebook with complete code and visualizations.
**Model:** distilbert/distilgpt2 enhanced with a value head for RL.
**Reward Model:** thanawan/bert-base-uncased-finetuned-humordetection
**Dataset:** CreativeLang/ColBERT_Humor_Detection

**Key Components**
GPT-2 PPO Fine-Tuning: We use trl's PPOTrainer to fine-tune GPT-2 Medium based on feedback from the humor classifier.
Custom Reward Function: Texts are scored using the BERT classifier trained on humor detection.
RLHF Training Loop: At each step, generated responses are evaluated for humor, and the model is updated accordingly.
Reward Tracking & Visualization: A line plot tracks reward improvements over PPO steps.
Custom Prompt Testing: You can input your own prompts at the end to generate humorous completions in real-time.

**📈 Results**
A reward comparison table highlights how responses improved before vs. after fine-tuning.
Average humor scores show a positive trend over time.
The model adapts to generate more light-hearted and funny outputs.


**Run the notebook:**

- Launch in Google Colab
- Open Humor_RLHF_GPT2.ipynb
- Run cells step-by-step

**Sample Prompts to Try**

- I told my boss I needed a day off because...
- My dog started talking yesterday, and he said...
- If I had a dollar for every time...
