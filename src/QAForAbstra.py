# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:18:01 2023

@author: kamblrah
"""

# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from typing import List, Tuple, Callable
import tkinter as tk
import os
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from openai import OpenAI
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

embeddings_path = "..\\EmbeddingsData\\abstra_embeddings.csv"

df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
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

# examples
openai.api_key = os.getenv("OPENAI_API_KEY")
strings, relatednesses = strings_ranked_by_relatedness("request name", df, top_n=5)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)
    
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'You are fist level smart suport about API queries. Remember input/output parameters for any request are multiple places you need to join it. Use the below api document to answer the subsequent question. If the answer cannot be found in the document, write "I could not find an answer."'
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
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
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

# def display_output():
#     input_text = entry.get()
#     respFromBot = ask(input_text)
#     label.config(text=f"{respFromBot}")

# # Create the main window
# root = tk.Tk()
# root.title("Simple UI")

# # Create a text entry widget
# entry = tk.Entry(root)
# entry.pack()

# # Create a button widget
# button = tk.Button(root, text="Submit", command=display_output)
# button.pack()

# # Create a label widget for displaying the output
# label = tk.Label(root, text="")
# label.pack()

# # Run the application
# root.mainloop()

# class ChatWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         # Set window properties
#         self.setGeometry(300, 300, 800, 350)
#         self.setWindowTitle('ChatBot UI')

#         # Create layout
#         layout = QVBoxLayout()

#         # Chat display area
#         self.chat_display = QTextEdit(self)
#         self.chat_display.setReadOnly(True)
        
#         # Set font for chat_display
#         chat_font = QFont("Tietoevry Sans 1 Medium", 20)
#         self.chat_display.setFont(chat_font)
        
#         layout.addWidget(self.chat_display)

#         # Input field
#         self.input_field = QLineEdit(self)
#         layout.addWidget(self.input_field)

#         # Submit button
#         submit_button = QPushButton('Submit', self)
#         submit_button.clicked.connect(self.submitText)
#         layout.addWidget(submit_button)

#         # Clear button
#         clear_button = QPushButton('Clear', self)
#         clear_button.clicked.connect(self.clearChat)
#         layout.addWidget(clear_button)

#         #key button
#         key_button = QPushButton('Get Key', self)
#         clear_button.clicked.connect(self.getKey)
#         layout.addWidget(key_button)

#         # Set the layout
#         self.setLayout(layout)
#     def submitText(self):
#         input_text = self.input_field.text()
#         self.chat_display.append(f"You: {input_text}")
#         self.input_field.clear()
    
#         # Create and start the ask thread
#         self.ask_thread = AskThread(input_text)
#         self.ask_thread.response_signal.connect(self.displayResponse)
#         self.ask_thread.start()

#     def displayResponse(self, response_text):
#         # Display the response in the chat
#         self.chat_display.append(f"ProAPIBot: {response_text}")
#     def clearChat(self):
#         # Clear the chat display area
#         self.chat_display.clear()

#     def getKey(self):
#         #get key file
#         fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.txt')
#         #f = open(fileName, "r")
#         #os.environ['OPENAI_API_KEY'] = f.read()
            
# class AskThread(QThread):
#     # Signal to send the response back to the main thread
#     response_signal = pyqtSignal(str)

#     def __init__(self, input_text):
#         super().__init__()
#         self.input_text = input_text

#     def run(self):
#         # Call the ask function and emit its response
#         response_text = ask(self.input_text)
#         self.response_signal.emit(response_text)
# app = QApplication(sys.argv)
# chat_window = ChatWindow()
# chat_window.show()
# sys.exit(app.exec_())

#ask('List all input parameters for RQ_ABS_CUST_GET?', model='gpt-4')
#ask('Explain use of IN_ACCOUNT_RISK_CLASS_LEVEL?', model='gpt-4')
#ask('What are input parameters for RQ_ABS_CUST_GET? ')
#ask('How to get customer position?')
#ask('How to add cash account to customer?')
#ask('What should my approach to list ledger balance per account group?')
#ask('Few accounts are displayed in red, I wonder why?')
#ask('Few accounts are displayed in red, I wonder why?', model='gpt-4')
#ask('Explain decision maker?')
#ask('Is it possible to get prices for instrument?')
#ask('How can I assure customer has position in given instrument?')
#ask('I am confused about IN_SEPERATOR. Help me to understand its usage', model='gpt-4')
#ask('I am confused about IN_SEPERATOR. Help me to understand its usage with example')
#ask('How can I place a order? tell me only required inputs for it', model='gpt-4')
#ask('List requests related to order management')
#ask('Summarise about model portfolio. Also mention corresponding APIs?')