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

# Load context document
context = """
The company's revenue grew by 20% in the third quarter, primarily driven by the launch of its new product line. 
However, operational costs increased by 15%, which offset some of the gains. The company expects stronger performance next quarter.
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
question = "How much was the company's revenue growth?"
answer = query_bot(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
