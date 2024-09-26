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
According to the court's decision, the defendant was found not guilty due to insufficient evidence linking them to the crime scene. 
The prosecution's case was primarily based on circumstantial evidence, which was not enough to secure a conviction.
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
question = "What was the court's decision?"
answer = query_bot(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
