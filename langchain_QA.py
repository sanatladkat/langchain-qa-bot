from langchain import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Load a pre-trained language model from Hugging Face
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Create a Langchain prompt template
template = """
You are an intelligent assistant. You will answer questions based on the following context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Load context document (simple text for this example)
context = """
Langchain is a framework designed to streamline the development of language model-driven applications. 
It allows users to create and manage workflows, connect language models to other tools, and deploy them in production environments.
"""

# Function to query the document using the Q&A model
def query_bot(question):
    inputs = {
        "question": question,
        "context": context
    }
    response = qa_model(inputs)
    return response['answer']

# Testing the Q&A bot
question = "What is Langchain?"
answer = query_bot(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
