#------------Step 1: Setup-------------

from dotenv import load_dotenv
import os
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

#----------Step 2: Load the Documents------------------

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

reader = SimpleDirectoryReader(docs_dir)
docs = reader.load_data()
print(f"Number of documents loaded: {len(docs)}")
print("Loaded the following files:")
for doc in docs:
    print(doc.metadata['file_name'])

#-------------Step 3: Build the Index and Query Engine---------------

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

#-------------Step 4: Query the Assistant-------------------------

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        text = node_with_score.node.get_content()
        text = " ".join(text.split())
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {text[:200]}...")
        print("-" * 30)

# The model sounds confident and actually answered all the questions correctly.
# Source documents are very helpful for this list of questions. 
# There were no surprises for me.

#-----------Step 5: Find a Failure--------------------

