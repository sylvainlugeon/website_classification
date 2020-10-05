import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup
import multiprocessing as mp

# can't group in a main because of gensim???

def main():
    print('loading model...')
    model = api.load('word2vec-google-news-300')
    
    a = model['hello']
    print(a)
    

if __name__ == "__main__":
    main()

    
