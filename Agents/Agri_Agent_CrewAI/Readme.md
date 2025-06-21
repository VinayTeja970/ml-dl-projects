
# Agriculture Support AI Agent using CrewAI and LLaMA 3.2

## Project Overview

This project involves building an AI-powered assistant to support farmers by providing real-time guidance on crop health, pest management, and fertilizer recommendations. The assistant was developed using **CrewAI**, integrated with a **locally hosted LLaMA 3.2 model** via **Ollama**, enabling cost-effective and offline inference.

---

## Features

- Provides insights on **crop diseases**, **pest control**, and **fertilizer use**
- Built with a **modular agent-based architecture** using CrewAI
- Supports **asynchronous task execution** for efficient query handling
- Uses **SerperDevTool** to access real-time agricultural information
- Hosted using **Ollama** to run LLaMA 3.2 locally without paid API dependence

---

## Key Components

### Agent Definition

- **Role:** Agriculture Assistant
- **Goal:** Offer expert guidance on farming practices and crop care
- **Backstory:** Acts as a university extension AI specialist aiding local farmers
- **LLM Used:** `ollama/llama3.2`

### Task Setup

- **Task Name:** Crop Disease Research Task
- **Description:** Researches crop diseases and returns treatment strategies
- **Execution:** Asynchronous for faster, concurrent operations

---

## Technologies Used

- Python
- CrewAI
- Ollama
- LLaMA 3.2
- LangChain (Ollama integration)
- SerperDevTool



## Author

**Vinay Teja Pathakamuri**  

