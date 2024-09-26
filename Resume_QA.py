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
Name: John Doe
Email: johndoe@example.com
Phone: +1-234-567-890
Education:
- Bachelor of Science in Computer Science, University of Example, 2018
- Master of Science in Data Science, University of Tech, 2020

Work Experience:
1. Data Scientist at ABC Corp (2020 - Present)
   - Built predictive models to improve customer retention by 30%
   - Led a team of 5 data analysts in developing machine learning pipelines
2. Junior Data Analyst at XYZ Ltd (2018 - 2020)
   - Developed dashboards for business insights using Python and Power BI
   - Automated data collection pipelines, reducing manual work by 50%

Skills:
- Programming: Python, SQL, R
- Machine Learning: Scikit-Learn, TensorFlow, PyTorch
- Data Visualization: Matplotlib, Seaborn, Power BI
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
question = "What is John's phone number?"
answer = query_bot(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
