from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
import os

documents = SimpleDirectoryReader('data/earnings').load_data()

STORAGE_DIR = "./storage"
def load_or_create_index(docs):
    index = None
    try:
        context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(context)
    except: 
        index = GPTVectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index
        
def ask(query):
    index = load_or_create_index(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(f"query> {query}")
    print(f"{response}")
    
ask("What is the targeted 2024 margin for Salesforce?")
ask("What is the total revenue for Q4 last quarter?")
