
<p align="center">
    <img src="./assets/logo.png" alt="Cucinabot logo" width="300">
</p>

# Cucinabot


[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A recipe suggestion agent built with **LangGraph** that allows you to interactively discover cooking ideas through a **Gradio** interface.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Getting Started](#getting-started)
- [Usage](#usage)  
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This repository implements a **LangGraph**‑based agent that helps you generate recipe suggestions. Users interact with the agent through a simple and intuitive **Gradio UI**, making it easy to input preferences, ingredients, or dietary constraints, and receive recipe ideas in response.

---

## Features

- 🧠 **Recipe Suggestion Agent**: Uses a language model within a LangGraph workflow to suggest recipes dynamically. A RAG system leveraging a [recipe dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images/versions/1) is added.
- 🖥️ **Gradio Interface**: Provides a browser-based, interactive UI for seamless user input and output display.  
- 🔄 **Stateful Interaction**: Leverages LangGraph connection to a Postgres database to maintain conversational context, improving suggestion relevance over multiple turns. It uses the same Postgres database to record name and food preferences of the user in a long-term memory store.

---

## Getting Started

To get started with **Cucinabot**, you have two options: using **Docker Compose** for an isolated environment or performing a **local installation** for more control.

---

### 🚀 Option 1: Docker Compose

Docker Compose simplifies the setup by managing dependencies and environment variables for you.

**Prerequisites:** [Docker](https://www.docker.com/products/docker-desktop) & [Docker Compose](https://docs.docker.com/compose/install/)

```bash
git clone https://github.com/pcumani/cucinabot.git
cd cucinabot
docker-compose up --build
```

### 🛠️ Option 2: Local Installation

For more flexibility or if you prefer not to use Docker, you can set up Cucinabot locally.

**Prerequisites:** Python 3.10 or higher, `pip` package manager, Virtual environment (optional but recommended), and a running PostgreSQL database (can be run via Docker).

1. Run PostgreSQL via Docker (example):

```bash
docker run -d --rm -e POSTGRES_USER='cucinabot' -e POSTGRES_PASSWORD='cucinabot' -p 5432:5432 postgres:15
```

2. Install and run Cucinabot locally:

```bash
git clone https://github.com/pcumani/cucinabot.git
cd cucinabot
# Optional : activate a virtual environment first
pip install -r requirements.txt
python app.py
```

---

## Usage

Once the application is running, open your browser and go to http://localhost:7860 to interact with Cucinabot. There you can enter your cooking preferences (e.g., ingredients, dietary restrictions) and receive personalized recipe suggestions powered by the LangGraph agent.

## Configuration
- system_prompt.yaml: Customize the agent’s initial behavior, tone, or style of suggestions using this system prompt file.

- API Keys or LLM Configuration: Depending on the model used (e.g., Google, HuggingFace), you may need to set environment variables like GOOGLE_API_KEY or equivalents, or modify code imports and parameters. You can add a .env file to the repository, containing all important environment variables.

## License
This project is licensed under the MIT License.
