from llama_index import GPTVectorStoreIndex, GPTListIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from collections import namedtuple
import os
import logging
logger = logging.getLogger(__name__)

_node_struct = namedtuple('NodeType', ['vector', 'list'])
NODE_TYPE = _node_struct('vector', 'list')

documents = SimpleDirectoryReader('data/earnings').load_data()

STORAGE_DIR = "./storage"
def load_or_create_index(docs, node_type):
    index = None
    try:
        context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(context)
    except: 
        parser = SimpleNodeParser() 
        nodes = parser.get_nodes_from_documents(docs)
        
        if node_type == NODE_TYPE.list:
            index = GPTListIndex(nodes, storage_context=context, name="listIndex")
        elif node_type == NODE_TYPE.vector:
            index = GPTVectorStoreIndex(nodes, storage_context=context, name="vectorIndex")
        else:
            raise TypeError(f"Not supported node type exception: {node_type}")
        
        #index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index
        
def ask(query):
    index = load_or_create_index(documents, NODE_TYPE.list)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(f"query> {query}")
    print(f"{response}")
    
ask("What is the targeted 2024 margin for Salesforce?")
ask("What is the total revenue for Q4 last quarter?")

while True:
    response = ask(input("Question: "))
    print(response)