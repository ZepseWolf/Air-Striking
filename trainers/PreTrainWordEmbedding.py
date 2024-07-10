import os
import sys
import pandas as pd
import fasttext
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from python_module import websiteModule

train_labels = []
colums_text = []
countPhishing=0
countLegit=0
print("start")
for i in range(1,300):
    mass_txt = ""
    file_path = '../scrapperBot/database/signatures/websites'+str(i)+'.json'
    with open(file_path, 'r') as file:
        # Load JSON data from the file
      
        for i , row in enumerate(json.load(file)):
            
            # json_data[i]['signatures'] = ' '.join(row['signatures'])
            current_classification = row['classification']
            
            if current_classification == 0 and countLegit < countPhishing:
                mass_txt += " ".join(row['websites'])
                colums_text.append(row['signatures'])
                train_labels.append(row['classification'])
                countLegit += 1
            elif current_classification == 1:
                mass_txt += " ".join(row['websites'])
                colums_text.append(row['signatures'])
                train_labels.append(row['classification'])
                countPhishing += 1

    with open("./mass.txt", 'a' , encoding='utf-8',errors= "ignore") as file:
        file.write(mass_txt)
        
model = fasttext.train_unsupervised(input="mass.txt",wordNgrams=2, dim=128, min_count=2000)
model.save_model("code_embedding_model.bin")


