import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests

from PIL import Image
import os
import glob

import pandas as pd
import numpy as np

import multiprocessing as mp


from tqdm import tqdm

def main():
    """ entry point """
    
    # file that contains the urls
    folder = "/dlabdata1/lugeon/"
    file_name = "websites_1000_5cat"
    ext = ".gz"
    
    # where the screenshots will be saved
    out_folder = folder + file_name + "_screenshots/"
    
    print('Collecting screenshots from ' + file_name + '...')
    
    # remove all files in out folder
    files = glob.glob(out_folder + '*')
    for f in files:
        os.remove(f)

    # reading dataframe
    df = pd.read_csv(folder + file_name + ext, names = ['uid', 'url', 'cat0'], header=0)
    df = df.sample(frac=1, random_state=42)
    df = df.iloc[:500]
    

    # progess bar
    pbar = tqdm(total = df.shape[0])

    def update_pbar(item):
        pbar.update(chunk_size)
        
    chunk_size = 10
    
    nb_cpu = int(mp.cpu_count() / 2)
    #nb_cpu = 1
    pool = mp.Pool(nb_cpu)

    # distribute the works within the workers
    for i in range(0, df.shape[0], chunk_size):
        job = pool.apply_async(worker, args=(i, i+chunk_size, df, out_folder), callback=update_pbar)

    pool.close()
    pool.join()
    pbar.close()


def take_screenshot(url, out_path):
    """ take a screenshot of a website and save it under the outpath """

    # dimensions of the browser window
    in_width = 1920
    in_height = 1080

    # dimension reduction, after the screenshot has been taken
    down_factor = 2
    out_width = int(in_width / down_factor)
    out_height = int(in_height / down_factor)

    # jpeg quality
    quality = 85

    # driver
    options = webdriver.ChromeOptions()
    options.headless = True
    
    # not download files automatically
    prefs = {}
    prefs["download.prompt_for_download"] = True
    prefs["download.default_directory"] = "/dlabdata1/lugeon/downloads"
    options.add_experimental_option("prefs", prefs)
    
    try:
    
        driver = webdriver.Chrome('/home/lugeon/drivers/chromedriver', options=options)
        
        timeout = 10

        driver.set_page_load_timeout(timeout)
        driver.set_window_size(in_width, in_height)
        
        r = requests.head(url, timeout=timeout)
        
        
        # check that the html status code is valid and that the accessed resource is an html page 
        # (and not a file that will be downloaded)
        status_code = str(r.status_code)
        nice_status_code = status_code.startswith('2') or status_code.startswith('3')
        nice_content_type = r.headers['content-type'].startswith('text/html')
        
        if not(nice_status_code and nice_content_type):
            return

        # access the url, takes a screenshot (only in png)
        driver.get(url)
        time.sleep(1) # so that the website's elements are loaded
        driver.save_screenshot(out_path + '.png')

        # convert the png into a jpeg of lesser dimensions and quality
        img = Image.open(out_path + '.png')
        img = img.convert('RGB')
        img = img.resize((out_width, out_height), Image.ANTIALIAS)
        img.save(out_path + '.jpeg', optimize=True, quality=quality)
        os.remove(out_path + '.png')
        
    except Exception as e:
        #print(e)
        return

    finally:
        driver.quit()


def worker(start_id, end_id, df, out_folder):
    """ takes screenshots of websites that are within a slice of the dataframe """

    # in case the df length is not divisible by the step size
    if end_id > df.shape[0]:
        end_id = df.shape[0]

    sliced_df = df.iloc[start_id: end_id]
    
    sliced_df.apply(lambda row: take_screenshot(row.url, 
                                                out_folder + str(row.uid)), # names of screenshots are the uid
                                                axis=1)


if __name__ == '__main__':
    main()