import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup


model = api.load('word2vec-google-news-300')

def word2vec_avg(str):
    if str == '':
        return None
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
    return df_valid

def clean_html(body):
    try:
        soup = BeautifulSoup(body, 'html.parser')
        a = soup.get_text(' ').replace('\n', '').lower()
        b = re.sub(r"[^a-z ]+","", a)
        c = ' '.join(b.split())
        return c
    except:
        return ''
    

def create_w2v_avg(df):
    df_valid = clean_df(df)
    df_valid['emb'] = df_valid.apply(lambda x: word2vec_avg(clean_html(x.html)), axis=1)
    return df_valid[['emb', 'cat']]


chunksize = 1000
reader = pd.read_json("/dlabdata1/lugeon/websites_40000_5cat_html.json.gz", orient='records', lines=True, chunksize=chunksize)

df_main = pd.DataFrame([])

i = 0
for chunk in reader:
    df_chunk = pd.DataFrame(chunk)
    df_chunk = create_w2v_avg(df_chunk)
    df_main = pd.concat((df_main, df_chunk))
    i +=1 
    print('{} embeddings done'.format(chunksize*i))
    
df_main.to_csv('/dlabdata1/lugeon/test.gz', compression='gzip')
    
