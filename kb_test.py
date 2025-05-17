"""
Test script for the vector-based knowledge base
"""
import os
import shutil
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(".env.local")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Add OPENAI_API_KEY to your .env.local file.")
    exit(1)

# Import the knowledge base
from knowledge_base import KnowledgeBase

# Define vector store directory
vector_store_dir = "data/vectorstore"

# Flag to track if we need to create a new vector store
create_new = not os.path.exists(vector_store_dir)
choice = 'r'  # Default to reuse if exists

# Check if vector store already exists
if not create_new:
    print(f"Vector store already exists at {vector_store_dir}")
    choice = input("Do you want to (r)euse existing store, (d)elete and recreate, or (q)uit? [r/d/q]: ").lower()
    
    if choice == 'q':
        print("Exiting...")
        exit(0)
    elif choice == 'd':
        print(f"Deleting existing vector store at {vector_store_dir}")
        shutil.rmtree(vector_store_dir)
        create_new = True
    else:
        print(f"Reusing existing vector store at {vector_store_dir}")

# Create knowledge base
kb = KnowledgeBase(persist_directory=vector_store_dir)

# If vector store doesn't exist or was deleted, add the PDF
if create_new:
    print("\nAdding PDF to knowledge base...")
    kb.add_pdf("data/electromech.pdf")
    print("Knowledge base created and stored in vector database!")
else:
    # Try to load existing vector store
    print("\nLoading existing knowledge base...")
    if kb.load():
        print("Knowledge base loaded successfully!")
    else:
        print("Failed to load knowledge base. Creating new one...")
        kb.add_pdf("data/electromech.pdf")

# Test with a few sample queries
sample_queries = [
    "What are the key features?",
    "What industries is this used in?",
    "What is the weight capacity?"
]

for query in sample_queries:
    print(f"\n\nQuery: {query}")
    results = kb.search(query)
    
    print(f"Found {len(results)} relevant chunks:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(result)

print("\nTest completed!")

# Interactive mode
print("\nEntering interactive mode. Type 'q' to quit.")
while True:
    query = input("\nEnter your query: ")
    if query.lower() == 'q':
        break
        
    results = kb.search(query)
    
    print(f"Found {len(results)} relevant chunks:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(result)
