import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup
import multiprocessing as mp

from transformers import BertTokenizer, BertModel
import torch

import sys


def main():
    """ Entry point """
    
    folder = "/dlabdata1/lugeon/"
    name = "websites_1000_5cat"
    ext = "_html.json.gz"

    print('computing embeddings for ' + name + ext)
    
    chunksize = 100
    reader = pd.read_json(folder + name + ext, orient='records', lines=True, chunksize=chunksize) #nrows=100

    df_main = pd.DataFrame([])
    nb_cpu = int(mp.cpu_count() / 2)
    #nb_cpu = 4
    pool = mp.Pool(nb_cpu)

    i = 0
    for chunk in reader:
        df_chunk = pd.DataFrame(chunk)
        df_chunk = clean_df(df_chunk)
        df_chunk = create_bert_embeddings(df_chunk, pool)
        df_main = pd.concat((df_main, df_chunk))
        i +=1 
        print('chunk / {} embeddings'.format(chunksize*i))
        
    pool.close()
    pool.join()

    out_path = folder + name + "_emb.gz"
    print('writing dataframe in ' + out_path)
    df_main.to_csv(out_path, compression='gzip')

    
def clean_df(df):
    """ Remove all none elements of a dataframe """
    
    df_valid = df[df.errcode == 200]
    df_valid = df_valid[df_valid.html.notnull()]
    
    return df_valid





# ************************************ W2V ************************************

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
    return word2vec_avg(body_clean)

def create_w2v_embeddings(df, pool):
    """ Compute and returns the embeddings of html bodies using multiprocessing """
    df['emb'] = pool.map(clean_and_embed, df.html)
    return df[['uid', 'emb', 'cat']]

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

    
    

# ************************************ BERT ************************************


def bert_clean_sentence(s):
    
    s = re.sub(r"’", "'", s) # transform the french ' into english ones
    
    match = r"[^a-zA-z\u00C0-\u00FF '’-]" # match anything that is not a letter (incl. accentued), space, apostrophe and dash 
    match += "|" + "(\W\-|\-\W)+" # match any dash that is not between letters
    match += "|" + "(\W'|'\W)+" # match any apostrophe that is not between letters
    
    s = re.sub(match, "", s) # remove the matches characters
    s = re.sub(r"\s+"," ", s) # replace any whitespace with a space
    s = re.sub(r" +"," ", s) # remove any sucession of spaces
    s = s.strip() # trim the final sentence
    
    return s


def bert_split(body):
    
    soup = BeautifulSoup(body, 'html.parser')
    a = soup.get_text('[SEP]').split('[SEP]')
    b = [s.split('.') for s in a if len(s) != 0]
    flat_b = [bert_clean_sentence(s) for sublist in b for s in sublist]
    trimmed_b = [s for s in flat_b if len(s) != 0]
    
    return trimmed_b

def bert_embed_sentence(text):
    
    tokenized_text = tokenizer.tokenize(text)
    
    if len(tokenized_text) > 510:
        tokenized_text = tokenized_text[:510]
        
    tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        output = bert_model(tokens_tensor, segments_tensors)
        hidden_states = output[2] # tensor of shape 
    
    token_states = torch.stack(hidden_states, dim=0)
    token_states = torch.squeeze(token_states, dim=1) 
    token_embeddings = token_states[-2] 
    
    return torch.mean(token_embeddings, dim=0)

def bert_embed_html(body):
    
    sentences = bert_split(body)
    l = len(sentences)
    
    acc = np.zeros(nb_dim)
    
    for s in sentences:
        acc += bert_embed_sentence(s).numpy()
        
    if l != 0:
        return (acc / len(sentences)).tolist()
    else:
        None

def create_bert_embeddings(df, pool):
    """ Compute and returns the embeddings of html bodies using multiprocessing """
    
    df['emb'] = pool.map(bert_embed_html, df.html)
    return df[['uid', 'emb', 'cat']]





# ************************************ ENTRY POINT ************************************

if __name__ == '__main__':
    
    args = sys.argv
    
    if args[1] == 'w2v':
        print('loading w2v model...')
        model = api.load('word2vec-google-news-300')
        
        main()
    
    elif args[1] == 'bert':
        
        print('loading bert model...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
        bert_model.eval()
        
        nb_dim = 768
        
        main()
        
    else:
        print('error')


    
