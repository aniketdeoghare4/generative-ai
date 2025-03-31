# imports
import ast
import pandas as pd
import tiktoken 
from scipy import spatial
from typing import List, Tuple, Callable
from bs4 import BeautifulSoup
import pandas as pd
import openai
import xml.etree.ElementTree as ET
import os
from sqlalchemy import create_engine, inspect
from pathlib import Path
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
import customtkinter as ctk
from tkinter import filedialog, StringVar, messagebox
import os
import threading
import requests
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4-turbo"


#prepare_sqlitedb_from_csv_xlsx.py
class PrepareSQLFromTabularData:
    """
    A class that prepares a SQL database from CSV or XLSX files within a specified directory.

    This class reads each file, converts the data to a DataFrame, and then
    stores it as a table in a SQLite database, which is specified by the application configuration.
    """
    def __init__(self, files_dir) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            files_dir (str): The directory containing the CSV or XLSX files to be converted to SQL tables.
        """
        self.files_directory = files_dir
        self.file_dir_list = os.listdir(files_dir)
        db_path = "./data/" + Path(self.files_directory).name + ".db"
        if(os.path.exists(db_path)):
            print("Deleting existing database '" + db_path + "'")
            os.remove(db_path)
        if(os.path.exists(db_path) is False):
            db_path = f"sqlite:///{db_path}"
            self.engine = create_engine(db_path)
            self.db_names_list = [os.path.splitext(file)[0] for file in self.file_dir_list]
            print("Create databases: ",self.db_names_list)
        else:
            raise RuntimeError("Resource in use")

    def _prepare_db(self):
        """
        Private method to convert CSV/XLSX files from the specified directory into SQL tables.

        Each file's name (excluding the extension) is used as the table name.
        The data is saved into the SQLite database referenced by the engine attribute.
        """
        for file in self.file_dir_list:
            full_file_path = os.path.join(self.files_directory, file)
            file_name, file_extension = os.path.splitext(file)
            if file_extension == ".csv":
                df = pd.read_csv(full_file_path)
            elif file_extension == ".xlsx" or file_extension == ".xls":
                if(file_extension == ".xlsx"):
                    df = pd.read_excel(full_file_path, engine="openpyxl")
                else:
                    df = pd.read_excel(full_file_path, engine="xlrd")
            else:
                raise ValueError("The selected file type is not supported")
            df.to_sql(file_name, self.engine, index=False)
            print("Database '" + file_name + "'added")
        print("")

    def _validate_db(self):
        """
        Private method to validate the tables stored in the SQL database.

        It prints out all available table names in the created SQLite database
        to confirm that the tables have been successfully created.
        """
        insp = inspect(self.engine)
        table_names = insp.get_table_names()

    def run_pipeline(self):
        """
        Public method to run the data import pipeline, which includes preparing the database
        and validating the created tables. It is the main entry point for converting files
        to SQL tables and confirming their creation.
        """
        self._prepare_db()
        self._validate_db()


#rag_retriever.py
class RAGRetriever:

    query_engine : any
    CONTEXT = '''You are an AI-powered assistant that provides accurate and relevant answers based on structured data.  
Your responses should be clear, concise, and natural, without mentioning any technical details about databases, SQL queries, or internal processing.  

### **Response Guidelines:**
1. **Provide Direct and Meaningful Answers:**  
   - If the answer is unclear or incomplete, respond with:
     In such cases just say *"I couldn't find precise information for your question." And also mention all the database names referred to*

2. **Interpret User Intent Accurately:**  
   - Understand the user's query and determine the most relevant data.  
   - Recognize synonyms, abbreviations, or related terms and match them to the correct information.  
   - If a column name contains spaces, ensure it is encapsulated in backticks like `` or square brackets "[]" when referring to it internally.  
   - Ensure responses are user-friendly and conversational.

Your role is to assist users by delivering **reliable, easy-to-understand responses** while keeping technical implementation details hidden.
'''

    def train_model(self, documentDirectory, api_key):
        sqlDb = PrepareSQLFromTabularData(documentDirectory)
        sqlDb.run_pipeline()
        sql_database = SQLDatabase(sqlDb.engine, include_tables=sqlDb.db_names_list)
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_obj = [SQLTableSchema(table_name=table) for table in sqlDb.db_names_list]
        obj_index = ObjectIndex.from_objects(table_schema_obj, table_node_mapping, VectorStoreIndex)
        llm = OpenAI(model="gpt-4-turbo", api_key=api_key)
        llm.system_prompt = self.CONTEXT
        self.query_engine = SQLTableRetrieverQueryEngine(sql_database, 
                                                        obj_index.as_retriever(similarity_top_k=3, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]),
                                                        llm=llm)
        #self.query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=sqlDb.db_names_list, llm=llm)
        return self.query_engine is not None

    def get_response(self, question):
        if hasattr(self, "query_engine"):
            print("User: ", question)
            response = self.query_engine.query(question)
            print("GPT: ", response)
        else:
            print("Query engine not initialized")
            raise("Query engine not initialized")
        return str(response)


#CreateEmbeddings.py
def create_embeddings(chunked_texts, model_name):
    response = openai.Embedding.create(model=model_name, input=chunked_texts)
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings

def process_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Get the textual content from the XML file
    text = ET.tostring(root, encoding='utf-8').decode('utf-8')
    return text

def save_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def create_embeddings(api_key):
    client = openai.OpenAI(
    api_key=api_key,  # this is also the default, it can be omitted
    )
    openai.api_key = api_key
    print("CWD: ", os.getcwd())
    xml_files_dir = ".\\datasets\\EmbeddingsInputFiles\\Xmls\\"
    csv_path = ".\\data\\EmbeddingsData\\RequestDescriptionsEmbddings.csv"
    EMBEDDING_MODEL = "text-embedding-ada-002"

    all_texts = []
    for file_name in os.listdir(xml_files_dir):
        if not file_name.endswith('.xml'):
            continue
        file_path = os.path.join(xml_files_dir, file_name)
        text = process_xml(file_path)
        all_texts.append(text)

    # Due to the limitation of OpenAI's API, we send a batch of up to 1000 texts at a time
    BATCH_SIZE = 1000
    embeddings = []
    print(len(all_texts[0]))
    for all_text in all_texts:
        print(len(all_text))    
        for batch_start in range(0, len(all_text), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = all_text[batch_start:batch_end]
            print(f"Batch {batch_start} to {batch_end-1}")
            print(len(batch))
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    # Save embeddings to csv
    save_to_csv(embeddings, csv_path)

#HtmlFileEmbeddings.py
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def load_and_prepare_html(file_path):
    # Open and read the HTML file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_text_from_given_classname(html, class_name):
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(str(html), 'html.parser')
    # Find all h3 tags with an id
    tags = soup.find_all(lambda tag: class_name in tag.get('class', []))    
    return tags

def extract_all_other_info(html):
    returnTable = extract_text_from_given_classname(html,'Return')
    description = extract_text_from_given_classname(html,'Description')
    inputTable = extract_text_from_given_classname(html,'Input')
    outputTable = extract_text_from_given_classname(html,'Output')
    return returnTable,description,inputTable,outputTable
def extract_text(html):
    soup = BeautifulSoup(str(html), 'html.parser')
    text = soup.get_text()
    return text
def extract_first_h3(html):    
    soup = BeautifulSoup(str(html), 'html.parser')    
    # Find the first h3 tag
    h3_tag = soup.find('h3')    
    # If h3 tag exists, get its text, otherwise return None
    h3_content = h3_tag.get_text() if h3_tag else None    
    return h3_content

def list_request_names(allRequests):
    all_texts = []
    for request in allRequests:
        requestName=extract_first_h3(request)
        otherInfo = extract_all_other_info(request)
        otherData = extract_text(otherInfo[1])
        noOfTokens = num_tokens(requestName + otherData)
        if noOfTokens > 1600:
            print(extract_first_h3(request))
        else:
            if(noOfTokens> 1600):
                print('Bluff1')
            all_texts.append(requestName+otherData)
            
        otherData = extract_text(otherInfo[2])
        noOfTokens = num_tokens(requestName + otherData)
        if noOfTokens > 1600:
            reminder = noOfTokens % 1550
            if reminder > 0:
                reminder = 1
            noOfSplits = (noOfTokens // 1550) + reminder
            splitData = split_html_rows(otherInfo[2],noOfSplits)
            for data in splitData:
                noOfTokens = num_tokens(requestName + extract_text(data))
                if(noOfTokens> 1600):
                     print('Bluff2')
                else:
                    all_texts.append(requestName+extract_text(data))
        else:
            noOfTokens = num_tokens(requestName + otherData)
            if(noOfTokens> 1600):
                print('Bluff3')
            all_texts.append(requestName+otherData)
            
        otherData = extract_text(otherInfo[3])
        noOfTokens = num_tokens(requestName + otherData)
        if noOfTokens > 1600:
            reminder = noOfTokens % 1550
            if reminder > 0:
                reminder = 1
            noOfSplits = (noOfTokens // 1550) + reminder
            splitData = split_html_rows(otherInfo[2],noOfSplits)
            for data in splitData:
                 noOfTokens = num_tokens(requestName + extract_text(data))
                 if(noOfTokens> 1600):
                     print('Bluff4')
                 else:
                     all_texts.append(requestName+extract_text(data))
        else:
            noOfTokens = num_tokens(requestName + otherData)
            if(noOfTokens> 1600):
                print('Bluff5')
            all_texts.append(requestName+otherData)      
    return  all_texts

def process_document(allRequests):
    all_texts = []
    for request in allRequests:
        requestName=extract_first_h3(request)
        otherInfo = extract_all_other_info(request)
        all_texts.append(get_request_descr(otherInfo, requestName))
        all_texts.extend(get_request_input_parameters(otherInfo, requestName))
        all_texts.extend(get_request_output_parameters(otherInfo, requestName))
    return all_texts

def get_request_descr(requestDetails, requestName):
    descr = extract_text(requestDetails[1])
    return 'Overview of '+ requestName + ': ' + descr
def get_request_input_parameters(requestDetails, requestName):
    return get_request_parameters(requestDetails[2],requestName,'Input')
def get_request_output_parameters(requestDetails, requestName):
    return get_request_parameters(requestDetails[3],requestName,'Output')
    
def get_request_parameters(requestDetails, requestName, preffix):
    # Initialize a BeautifulSoup object with the provided HTML
    soup = BeautifulSoup(str(requestDetails), 'html.parser')    
    # Find all tr (table row) tags
    tr_tags = soup.find_all('tr')
    totalTokens = 0 
    allLines=[]
    strInputParam = preffix + ' parameters for ' + requestName +' are '
    strInputParamNew = ''
    for tr_tag in tr_tags:        
        td_tags = tr_tag.find_all('td')
        if len(td_tags) > 0:
            strInputParamNew = ''
            strInputParamNew += 'Name:' + extract_text(td_tags[1]) + ' '
            strInputParamNew += ('Description:' + extract_text(td_tags[3]) + ' ')
            strInputParamNew += ('Mandatory:'+ ('Y' if len(extract_text(td_tags[2])) > 0 else 'N' ) + ' ')
            strInputParamNew += ('DataDict type:' + extract_text(td_tags[4]) + ' ')
            strInputParamNew += ('Database type:' + extract_text(td_tags[5]) + ' ')
            totalTokens += num_tokens(strInputParamNew)
            if totalTokens > 1550:                
                allLines.append(strInputParam + strInputParamNew)
                strInputParam = preffix + ' parameters for ' + requestName +' are '
                totalTokens = 0
            else:
                strInputParam += strInputParamNew
    allLines.append(strInputParam)      
    return allLines
def split_html_rows(html, num_chunks):
    # Initialize a BeautifulSoup object with the provided HTML
    soup = BeautifulSoup(str(html), 'html.parser')
    
    # Find all tr (table row) tags
    tr_tags = soup.find_all('tr')
    
    # Determine the size of each chunk
    chunk_size = len(tr_tags) // num_chunks
    remainder = len(tr_tags) % num_chunks
    
    chunks = []
    i = 0
    
    # Split the rows into chunks
    for _ in range(num_chunks):
        end = i + chunk_size + (1 if remainder > 0 else 0)
        chunks.append(tr_tags[i:end])
        i = end
        remainder -= 1
    
    return chunks      

def initialize_html_embeddings(api_key):
    html = load_and_prepare_html(".\\datasets\\EmbeddingsInputFiles\\Abstra.html")
    allRequests = extract_text_from_given_classname(html, "Request")
    allTexts = process_document(allRequests)
    for text in allTexts:
        if num_tokens(text) > 1600:
            print('Bluff')
            print(num_tokens(text))
    # calculate embeddings
    process_document(allRequests)
    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    embeddings = []


    client = openai.OpenAI(
    api_key=api_key,  # this is also the default, it can be omitted
    )

    openai.api_key = api_key
    for batch_start in range(0, len(allTexts), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = allTexts[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": allTexts, "embedding": embeddings})   

    SAVE_PATH = ".\\data\\EmbeddingsData\\abstra_embeddings.csv"

    df.to_csv(SAVE_PATH, index=False)
        
    myData = extract_all_other_info(allRequests[0])
    print(myData[2])
    rowChunks = split_html_rows(myData[2],3)
    print(num_tokens( extract_text(rowChunks[1])))
    print(num_tokens(allTexts[0]))
    print(extract_text(rowChunks[0]))

#QAForAbstra.py
def initialize_abstra_qa(api_key):
    client = openai.OpenAI(api_key=api_key)
    embeddings_path = ".\\data\\EmbeddingsData\\abstra_embeddings.csv"

    df = pd.read_csv(embeddings_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    strings, relatednesses = strings_ranked_by_relatedness("request name", df, client, top_n=5)
    # for string, relatedness in zip(strings, relatednesses):
    #     print(f"{relatedness=:.3f}")
    #     print(string)
    return df, client;

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    client: openai.OpenAI,
    relatedness_fn: Callable[[list, list], float] = lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> Tuple[List[str], List[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
    
# def num_tokens(text: str, model: str = GPT_MODEL) -> int:
#     """Return the number of tokens in a string."""
#     encoding = tiktoken.encoding_for_model(model)
#     return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    client: openai.OpenAI,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, client)
    introduction = 'You are fist level smart support about API queries. Remember input/output parameters for any request are multiple places you need to join it. Use the below api document to answer the subsequent question. If the answer cannot be found in the document, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nRequest document:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame,
    client: openai.OpenAI,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    response_message: str = ""
):
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, client, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the request document."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    
    response_message = response.choices[0].message.content
    return response_message

#UI Files Region
#ChatBot.py
class ChatBotDialog(ctk.CTk):
    RAG_CSV_XL = "RAG .csv/.xlsx"
    ABSTRA = "Abstra Documents QA"
    rag = RAGRetriever()
    rag_ready = False
    def __init__(self, api_key):
        super().__init__()

        self.api_key = api_key
        self.title("Sample Dialog")
        self.geometry("600x700")  # Fixed size
        self.resizable(False, False)  # Make the size fixed

        # Dropdown Menu
        self.option_var = StringVar(value="")
        self.dropdown = ctk.CTkOptionMenu(self, variable=self.option_var, values=[self.RAG_CSV_XL, self.ABSTRA], command=self.selection_changed)
        self.dropdown.pack(pady=10)

        # Directory input and load button
        self.dir_label = ctk.CTkLabel(self, text="Select Directory:")
        self.dir_label.pack_forget()
        #self.dir_label.pack(pady=10)

        self.dir_button = ctk.CTkButton(self, text="Browse", command=self.load_directory)
        self.dir_button.pack_forget()
        #self.dir_button.pack(pady=10)

        self.message_label = ctk.CTkLabel(self, text="", text_color="green")
        self.message_label.pack_forget()
        #self.message_label.pack(pady=5)

        # Chatbox UI
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.chat_input = ctk.CTkEntry(self, placeholder_text="Type a message...")
        self.chat_input.pack(fill='x', padx=10, pady=5)
        self.chat_input.bind("<Return>", self.send_message)
        self.chat_input.configure(state="disabled")

    def selection_changed(self, selected_value):
        
        for widget in self.chat_frame.winfo_children():
            widget.pack_forget()
            
        if(selected_value == self.ABSTRA):
            self.dir_button.pack_forget()
            self.message_label.pack_forget()
            self.dir_label.pack_forget()
            if(messagebox.askyesno("Message", "Do you want to refresh/create embeddings?")):
                create_embeddings(self.api_key)
                initialize_html_embeddings(self.api_key)
            self.df, self.client = initialize_abstra_qa(self.api_key)
            self.chat_input.configure(state="normal")
            self.chat_input.focus()

        elif(selected_value == self.RAG_CSV_XL):
            self.dir_label.pack(pady=10, after=self.dropdown)
            self.dir_button.pack(pady=10, after=self.dir_label)
            self.message_label.pack(pady=5, after=self.dir_button)
            if(self.rag_ready):
                self.chat_input.configure(state="normal")
            else:
                self.chat_input.configure(state="disabled")

    def load_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.message_label.configure(text=f"Directory Loaded: {directory}", text_color="green")
            absolute_path = os.path.abspath(directory)
            self.rag_ready = self.rag.train_model(absolute_path, self.api_key)
            if(self.rag_ready):
                self.chat_input.configure(state="normal")
                self.chat_input.focus()
        else:
            self.message_label.configure(text="No directory selected", text_color="red")
            self.chat_input.configure(state="disabled")
        self.chat_input.focus()

    def get_response(self, user_message):
        response=""
        if(self.dropdown.get() == self.RAG_CSV_XL):
            response = self.rag.get_response(user_message)
        elif(self.dropdown.get() == self.ABSTRA):
            response = ask(user_message, self.df, self.client)
        self.chat_frame.after(0, self.display_message, response)

    def send_message(self, event=None):
        user_message = self.chat_input.get()
        if user_message.strip():
            self.display_message(user_message, align="right", bg_color="#D1E8FF")
            self.chat_input.delete(0, 'end')
            threading.Thread(target=self.get_response, args=(user_message,), daemon=True).start()

    def display_message(self, message, align="left", bg_color="#E8E8E8"):
        # Adjust textbox height dynamically based on content
        line_count = message.count('\n') + 1
        char_width = 40
        wrapped_lines = sum([len(line) // char_width + 1 for line in message.split('\n')])
        height = max(40, 13 * max(line_count, wrapped_lines))

        # Create a new textbox for each message
        textbox = ctk.CTkTextbox(self.chat_frame, height=height, width=450, wrap='word', fg_color=bg_color, text_color="black")
        textbox.insert('0.0', message)
        textbox.configure(state='disabled', font=("Arial", 12, "bold"))

        # Align based on sender
        if align == "right":
            textbox.pack(padx=10, pady=5, anchor='e')  # Align to right
        else:
            textbox.pack(padx=10, pady=5, anchor='w')  # Align to left


#APIKeyForm.py
ctk.set_appearance_mode("dark")
app = ctk.CTk()

def read_api_key(file_path):
    """Reads the OpenAI API key from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def validate_api_key(api_key):
    """Validates the OpenAI API key by making a test request."""
    openai.api_key = api_key
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return True, "API key is valid."
    else:
        return False, "Invalid API key"

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        api_key = read_api_key(file_path)
        if api_key is None:
            result_label.configure(text="Error while reading key from file.", text_color="red")
        else:
            is_valid, validation_result = validate_api_key(api_key)
            result_label.configure(text=validation_result, text_color="green" if is_valid else "red")
            if is_valid:
                app.destroy()
                print("Previous working directory: ", os.getcwd())
                os.chdir(".")
                print("Current working directory: ", os.getcwd())
                chatbot_app = ChatBotDialog(api_key)
                chatbot_app.mainloop() 

app.title("OpenAI API Key Validator")
app.geometry("400x200")

frame = ctk.CTkFrame(app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

browse_button = ctk.CTkButton(frame, text="Browse API Key File", command=browse_file)
browse_button.pack(pady=10)

result_label = ctk.CTkLabel(frame, text="Select a file to validate API key", wraplength=350)
result_label.pack(pady=10)

app.mainloop()