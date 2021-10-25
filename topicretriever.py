import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

#get abstract
def getabstract(file_path):
    print("Converting " + file_path + " to textfile")
    os.system("pdftotext {} {}".format(file_path, "temp.txt"))
    source = open("temp.txt", "r")
    text = source.readlines()
    source.close()
    result = ""
    start_read = False
    end_read = False

    for line in text:
        if line.find('Keywords:') != -1:
            break
        if start_read == True:
            result += line
        else:
            if line.find('ABSTRACT') != -1:
                start_read = True
    os.remove("temp.txt")
    calculatetfidf(file_path, result)

def calculatetfidf(filepath, dataset):
    print("Calculating " + filepath[:-4] + " tf-idf value")
    result_file = open("result/" + filepath[:-4] + "_tfidf.txt", "w")
    tfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words="english")
    tfIdf = tfIdfVectorizer.fit_transform([dataset])
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    df = df.drop('TF-IDF', axis=1)
    text = df.head(10).index.tolist()
    for line in text:
        result_file.write(line + "\n")
    result_file.close()

directory = "."
Path("result/").mkdir(exist_ok=True)
for filename in os.listdir(directory):
    if filename.find(".pdf") > -1:
        getabstract(filename)

print("All task finished")