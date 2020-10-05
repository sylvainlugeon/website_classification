import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup

import multiprocessing as mp


def main():
    """ Entry point """
    
    folder = "/dlabdata1/lugeon/"
    name = "websites_1000_5cat"
    ext = "_html.json.gz"

    print('computing embeddings for ' + name + ext)
    
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
    
    

def word2vec_avg(sentence):
    """ Compute the words embeddings in a sentence and average them """
    
    acc = np.zeros(300)
    counter = 0
    for word in sentence.split(' '):
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
    """ Remove all none elements of a dataframe """
    
    df_valid = df[df.errcode == 200]
    df_valid = df_valid[df_valid.html.notnull()]
    
    return df_valid



def clean_html(body):
    """ Parse the html body and returns only the words appearing on the website """
    
    try:
        soup = BeautifulSoup(body, 'html.parser')
        a = soup.get_text(' ').replace('\n', '').lower()
        b = re.sub(r"[^a-z ]+","", a)
        c = ' '.join(b.split())
        return c
    except:
        return ''
    

def clean_and_embed(body):
    """ Clean and compute the embedding of an html body """
    
    body_clean = clean_html(body)
    body_length = len(body_clean.split())
    
    return word2vec_avg(body_clean), body_length

def create_w2v_embeddings(df, pool):
    """ Compute and returns the embeddings of html bodies using multiprocessing """
    
    df['emb_n_len'] = pool.map(clean_and_embed, df.html)
    df['emb'] = df.apply(lambda x: x.emb_n_len[0], axis=1)
    df['len'] = df.apply(lambda x: x.emb_n_len[1], axis=1)
    return df[['emb', 'len', 'cat']]


if __name__ == '__main__':
    
    print('loading model...')
    model = api.load('word2vec-google-news-300')
    
    main()


    
