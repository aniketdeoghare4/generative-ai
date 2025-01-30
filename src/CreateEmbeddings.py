# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:35:58 2023

@author: kamblrah
"""
import os
import openai
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from openai import OpenAI

# OpenAI key setup
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
openai.api_key = os.environ['OPENAI_API_KEY']

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

# Assume we are processing the xml files in the xml_files_dir directory
xml_files_dir = "..\\EmbeddingsInputFiles\\Xmls\\"
csv_path = "..\\EmbeddingsData\\RequestDescriptionsEmbddings.csv"
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


