# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:12:28 2023

@author: kamblrah
"""

from bs4 import BeautifulSoup
import pandas as pd
import tiktoken  # for counting tokens
import openai  # for generating embeddings
import os
from openai import OpenAI

GPT_MODEL = "gpt-3.5-turbo"  


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
html = load_and_prepare_html("..\\EmbeddingsInputFiles\\Abstra.html")
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


client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

openai.api_key = os.environ['OPENAI_API_KEY']
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

SAVE_PATH = "..\\EmbeddingsData\\abstra_embeddings.csv"

df.to_csv(SAVE_PATH, index=False)
     
myData = extract_all_other_info(allRequests[0])
print(myData[2])
rowChunks = split_html_rows(myData[2],3)
print(num_tokens( extract_text(rowChunks[1])))
print(num_tokens(allTexts[0]))
print(extract_text(rowChunks[0]))

