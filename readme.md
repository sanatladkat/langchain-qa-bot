# Langchain Q&A Bot: Building Your First Intelligent Agent

For a detailed explanation of the code and more use cases, check out the full article on Medium:
[Build Your First Intelligent Agent: A Simple Q&A Bot](https://medium.com/@arsanatl/build-your-first-intelligent-agent-a-simple-q-a-bot-b41113d12155)

This repository contains the Python code for building a simple Q&A bot using Langchain and an open-source pre-trained language model from Hugging Face. The project is part of an article series on Langchain and Langgraph for agentic workflows with open-source LLMs.

## Overview

The goal of this project is to introduce the use of Langchain for creating agentic workflows with Large Language Models (LLMs). In this example, we build a Q&A bot that answers user queries based on a given context document. The bot uses a pre-trained model from Hugging Face to process the input and return relevant answers.

### Key Features:
- **Langchain Integration**: Orchestrates the interaction between user input and the language model.
- **Pre-trained LLM**: Uses a pre-trained Hugging Face model to process questions.
- **Customizable Context**: Modify the document context to make the bot answer queries about any dataset, including resumes, financial reports, research papers, and more.
- **Real-world Use Cases**: The bot can handle a wide range of use cases, from legal document parsing to medical insights extraction.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system.

### Running the Q&A Bot
1. Clone the repository:

```bash
git clone https://github.com/sanatladkat/langchain-qa-bot.git
cd langchain-qa-bot
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the python scripts:

```bash
python Resume_QA.py
python medical_QA.py
python legal_QA.py
python FinReport_QA.py
```

4. Modify the context variable in the code to change the document against which the bot will answer questions.

### Example Use Cases:
1. Summarizing Financial Reports: Automate the extraction of key insights from complex financial data.
2. Medical Paper Insights: Query important data points from medical research documents.
3. Resume Q&A Bot: Create an intelligent agent that answers questions about a CV or resume, ideal for HR or recruitment.
4. Legal Document Parsing: Extract decisions or key facts from legal documents.