import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup
import multiprocessing as mp

# can't group in a main because of gensim???

folder = "/dlabdata1/lugeon/"
name = "websites_40000_5cat"
ext = "_html.json.gz"

print('computing embeddings for ' + name + ext)

print('loading model...')
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
        return (acc / counter).tolist()
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
    
def clean_and_embed(str):
    return word2vec_avg(clean_html(str))
    

def create_w2v_embeddings(df, pool):
    df['emb'] = pool.map(clean_and_embed, df.html)
    return df[['emb', 'cat']]


chunksize = 1000
reader = pd.read_json(folder + name + ext, orient='records', lines=True, chunksize=chunksize)

df_main = pd.DataFrame([])
pool = mp.Pool(int( mp.cpu_count() / 2))

i = 0
for chunk in reader:
    df_chunk = pd.DataFrame(chunk)
    df_chunk = clean_df(df_chunk)
    df_chunk = create_w2v_embeddings(df_chunk, pool)
    df_main = pd.concat((df_main, df_chunk))
    i +=1 
    print('chunk / {} embeddings'.format(chunksize*i))

pool.close()
pool.join()

out_path = folder + name + "_emb.gz"
print('writing dataframe in ' + out_path)
df_main.to_csv(out_path, compression='gzip')


    