import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup


model = api.load('word2vec-google-news-300')

def word2vec_avg(str):
    acc = np.zeros(300)
    counter = 0
    for word in str.split(' '):
        try:
            vector = model[word]
            acc += vector
            counter += 1
        except:
            pass
    if counter != 0:
        return acc / counter
    else:
        return None

def clean_df(df):
    df_valid = df[df.html.notnull()]
    df_valid = df_valid[df_valid.apply(lambda x: x.html != '', axis=1)]
    return df_valid

def clean_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    a = soup.get_text(' ').replace('\n', '').lower()
    b = re.sub(r"[^a-z ]+","", a)
    c = ' '.join(b.split())
    return c

def create_w2v_avg(df):
    df_valid = clean_df(df)
    df_valid['emb'] = df_valid.apply(lambda x: word2vec_avg(clean_html(x.html)), axis=1)
    return df_valid[['emb', 'cat']]



reader = pd.read_json("../data/websites_1000_5cat_html.json.gz", orient='records', lines=True, chunksize=100)

df_main = pd.DataFrame([])

for chunk in reader:
    df_chunk = pd.DataFrame(chunk)
    df_chunk = create_w2v_avg(df_chunk)
    df_main = pd.concat((df_main, df_chunk))
    
df_main.to_csv('test.gz', compression='gzip')
    
