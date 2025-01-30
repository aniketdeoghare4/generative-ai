from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from prepare_sqlitedb_from_csv_xlsx import PrepareSQLFromTabularData
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine

#table_list = ["DERIVATIVES_ENXC", "DERIV_EURO_COMMISSION_SETUP", "TRADE_FLOW_STOCK_OPTIONS"];

def train_model(documentDirectory):
    load_dotenv()
    sqlDb = PrepareSQLFromTabularData(documentDirectory)
    sqlDb.run_pipeline()
    sql_database = SQLDatabase(sqlDb.engine, include_tables=sqlDb.db_names_list)
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=sqlDb.db_names_list, llm=llm, embed_model='local')

    return query_engine

def get_response(query_engine, question):
    response = query_engine.query(question)
    return str(response)

if __name__ == "__main__":
    query_engine = train_model("C:\\Users\\deoghani\\Documents\\Tasks\\AIComm\\RAG\\datasets\\ALM")
    #print(get_response(query_engine, "Please show the description of the issue with Key TESTAUT-990 in ledger"))
    #print(get_response(query_engine, "List all the issues that Shahare Vaibhav has worked on in ledger"))
    #print(get_response(query_engine, "List key of all the issues with highest priority in ledger and show the priority as well"))
    print(get_response(query_engine, "Please provide me the answers based only on given input database to you, otherwise simply say database is not sufficient to answer this question"))
    while(True):
        question = input("You: ")
        if question == 'x' or question == 'X':
            break;
        print("GPT: ",get_response(query_engine, question))
    del query_engine


# For later references
    
    # manually set extra context text
    # aut_ledger_text = (
    # "This table has list of all the issues along with the unique key Key\n"
    # "Issues mainly relate with the ledger and also has the Component Area which it has impacted on\n"
    # "The person who has worked on the task is mentioned as Asignee")
    
    # aut_abstra_text = (
    # "This table has list of all the issues along with the unique key Key\n"
    # "Issues are related with the abstra tests and also has the Component Area which it has impacted on\n"
    # "The person who has worked on the task is mentioned as Asignee")

    # table_node_mapping = SQLTableNodeMapping(sql_database)
    # table_schema_objs = [(SQLTableSchema(table_name=table_list[0], context_str=aut_abstra_text)),
    #                      (SQLTableSchema(table_name=table_list[1], context_str=aut_ledger_text))]
    
    # obj_index = ObjectIndex.from_objects(table_schema_objs,table_node_mapping,VectorStoreIndex)
    
    
    # query_engine = SQLTableRetrieverQueryEngine(sql_database, obj_index.as_retriever(similarity_top_k=1), llm=llm)
    
    #table_schema_objs = [SQLTableSchema(table_name="AUT_ABSTRA"), SQLTableSchema(table_name="AUT_LEDGER")]