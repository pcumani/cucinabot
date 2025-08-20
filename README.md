
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
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
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
- 🔄 **Stateful Interaction**: Leverages LangGraph connection to a Postgres database to maintain conversational context, improving suggestion relevance over multiple turns.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher  
- `pip` package manager  
- (Optional) Virtual environment (recommended)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/pcumani/langgraph_agent.git
    cd langgraph_agent
    ```

2. (Optional) Set up and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To start the Gradio-based agent:

```bash
python app.py

```

This will launch a local web interface (http://localhost:7860/), where you can enter your cooking preferences (e.g., ingredients, dietary restrictions) and receive personalized recipe suggestions powered by the LangGraph agent.

## Configuration
- system_prompt.yaml: Customize the agent’s initial behavior, tone, or style of suggestions using this system prompt file.

- API Keys or LLM Configuration: Depending on the model used (e.g., Google, HuggingFace), you may need to set environment variables like GOOGLE_API_KEY or equivalents, or modify code imports and parameters.

## License
This project is licensed under the MIT License.