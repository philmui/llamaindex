from llama_index import GPTVectorStoreIndex, GPTListIndex, SimpleDirectoryReader, \
                        LLMPredictor, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from langchain import OpenAI

from collections import namedtuple
import os
import logging
logger = logging.getLogger(__name__)

_node_struct = namedtuple('_node_struct', ['vector', 'list'])
NODE_TYPE = _node_struct('vector', 'list')

documents = SimpleDirectoryReader('data/earnings').load_data()

STORAGE_DIR = "./storage"
def load_or_create_index(docs, node_type):
    index = None
    try:
        context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(context)
    except: 
        max_input_size = 4096
        num_output = 256
        chunk_overlap_ratio = 0.1        
        prompt_helper = PromptHelper(context_window=max_input_size, 
                                     num_output=num_output,
                                     chunk_overlap_ratio=chunk_overlap_ratio)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model="text-davinci-003"))

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context) 
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
    query = input("Question: ")
    if "quit" not in query:
        response = ask(query)
        print(response)
    else: 
        exit()
    
