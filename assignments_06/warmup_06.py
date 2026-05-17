from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

#----------------RAG Concepts--------------

#----------Concepts Question 1-------------

# A. RAG is best for this scenario - the assistant will need to read the library 
# before answering. RAG will also allow it to keep up with the updates.

# B. This is a scenario for fine tuning. The model will need the materials that
# are not available online to meet the expectations. Unlike scenario A, the examples
# are not going to be updated.

# C. Prompt engineering is optimal here, since all the info the model needs 
# is going to be provided in one single prompt.

#----------Concepts Question 2------------

# A wrong answer given in a confident tone is more harmful because it creates a 
# false impression the model has done the fact checking and we have nothing to worry 
# about. Earlier in AI days a chatbot gave me a wrong confident reply about a medication
# dosage and it almost cost me, so I'm usually fact checking when asking about
# important stuff.

#---------Concepts Question 3-------------

# steps = [
#     "Extract text from source documents", -- receive data
#     "Split text into chunks", -- start processing
#     "Convert text chunks into embeddings", -- convert into digital data
#     "Receive the user's query", -- get the actual task
#     "Embed the user's query", -- transform the query
#     "Retrieve the most relevant chunks", -- find matching data
#     "Inject retrieved chunks into the prompt", -- create a prompt based on the data
#     "Generate a response from the LLM" - complete the task
# ]

#----------------Keyword RAG-----------------------

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
#------------------Keyword Question 1------------------

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query, documents)

# With verbose set to True, the model selected the document 'loyalty' that had an 
# overlapping word 'your', which is a technical match but is not relevant to the 
# meaning of the question. It didn't recognize 'your' as a stop word since it's
# not equal to 'our'. 'Weekend' wasn't selected because it's not an exact
# match to 'weekends'.

#----------------Keyword Question 2-----------------

query = "Do you have anything without caffeine?"

simple_keyword_retrieval(query, documents)

# Simple keyword match failed again. The model wasn't able to select a document 
# because non of the selected tokens had a match in the text.
# Semantic retrieval would be more efficient, because it relies on meaning
# and not exact words.

#-----------------Keyword Question 3----------------

query = "How do I sign up for rewards?"

# Prediction: no documents will be found because not of the token will have
#  exact matches in the documents.

simple_keyword_retrieval(query, documents)

#---------------Semantic RAG Concepts---------------

#---------------Semantic Question 1-----------------

# 1. Vector embedding is transforming text, image, audio etc. into its numeric
# equivalent. 
# 2. 0.85 chunk is more relevant and the number tells me the text can contain 
# information that is relevant to the query.
# 3. Semantic search relies on the meaning rather than exact words, so 
# it can find relevant chunk even with no matches.

#---------------Semantic Question 2------------------

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Close meaning|
# | What is retrieved?         | Full document                     | Text chunks  |
# | Can it handle synonyms?    | No                                | Yes          |
# | Storage format             | Plain text dictionary             | Vector embeddings|
# | Relevance score            | Number of overlapping keywords    | Context score|

#----------------LlamaIndex------------------

#------------LlamaIndex Question 1------------

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Import and use of pdf reader suggested by AI because pdf files were not 
# parse correctly as is. Also log suprression was suggested by AI.

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from llama_index.readers.file import PDFReader
parser = PDFReader()
docs = SimpleDirectoryReader(
    "assignments_06/resources/brightleaf_pdfs",
    file_extractor={".pdf": parser}
).load_data()
index = VectorStoreIndex.from_documents(docs)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        text = node_with_score.node.get_content()
        text = " ".join(text.split())
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {text[:150]}...")
        print("-" * 30)

# For each question, only the first retrieved chunk had relevant information.
# Model sounds confident about the info it retrieved. Neutral informative tone.
# Only chunks 2 and 3 were unexpected.

#--------------LlamaIndex Question 2-------------

query_engine = index.as_query_engine(similarity_top_k=1)

print(f"\nQ: {questions[0]}")
response = query_engine.query(questions[0])
print("A:", response)
    
for node_with_score in response.source_nodes:
    text = node_with_score.node.get_content()
    text = " ".join(text.split())
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {text[:150]}...")
    print("-" * 30)

query_engine = index.as_query_engine(similarity_top_k=5)

print(f"\nQ: {questions[0]}")
response = query_engine.query(questions[0])
print("A:", response)
    
for node_with_score in response.source_nodes:
    text = node_with_score.node.get_content()
    text = " ".join(text.split())
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {text[:150]}...")
    print("-" * 30)

# Both responses provide relevant and correct information. The response based on 
# top_k=5 is slightly longer but it doesn't provide any visible benefits over response
# one. Only one document in the folder is relevant to this question.

#-----------------LlamaIndex Question 3----------------

query_engine = index.as_query_engine(similarity_top_k=3)

question_3 = "When's Brightleaf expected to reach its financial peak?"
print(f"\nQ: {question_3}")
response = query_engine.query(question_3)
print("A:", response)
    
for node_with_score in response.source_nodes:
    text = node_with_score.node.get_content()
    text = " ".join(text.split())
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {text[:150]}...")
    print("-" * 30)

# I expected the model to reply there's not enough data to make predictions.
# Instead, it just used the past numbers that were available. To improve model's
# performance I'd tell it in the prompt to pay attention it to the timeframe 
# and clearly distinguish if a question is about the past, present or future.

#--------------LlamaIndex Question 4-------------------

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"

response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

q2 = "Are we ever going to run out of solar energy?"
response2 = query_engine.query(q2)
print(response2)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q2, response=response2)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q2, response=response2)
print("Relevancy Result: " + str(relevancy_result.score))

# Faithfulness 1.0 = answer grounded in retrieved chunks. 0.0 = contradicts context.
# Relevancy = were the right chunks retrieved, not whether the answer is correct.
# q2 scored 1.0 on both despite the answer coming from model's general knowledge,
# not the documents — showing LLM-as-a-judge can be fooled.
# Simple accuracy metrics don't work for RAG — answers are free-form, not right/wrong.