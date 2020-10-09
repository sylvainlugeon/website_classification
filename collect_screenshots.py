import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from PIL import Image
import os

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
    out_folder = folder + file_name + "_screenshots_upper/"
    
    print('Collecting screenshots from ' + file_name + '...')

    # reading dataframe
    df = pd.read_csv(folder + file_name + ext, names = ['uid', 'url', 'cat0'], header=0)
    df = df.iloc[:100]

    # progess bar
    pbar = tqdm(total = df.shape[0])

    def update_pbar(item):
        pbar.update(chunk_size)
        
    chunk_size = 10
    
    nb_cpu = int(mp.cpu_count() / 4)
    pool = mp.Pool(nb_cpu)

    for i in range(0, df.shape[0], chunk_size):
        job = pool.apply_async(worker, args=(i, i+chunk_size, df, out_folder), callback=update_pbar)

    pool.close()
    pool.join()
    pbar.close()


def take_screenshot(url, out_path):
    """ take a screenshot of a website and save it under the outpath """

    # dimensions of the browser window, need to be tuned!
    in_width = 1920
    in_height = 2 * 1080

    # dimension reduction, after the screenshot has been taken
    down_factor = 2
    out_width = int(in_width / down_factor)
    out_height = int(in_height / down_factor)

    # jpeg quality
    quality = 85

    # if the file already exists, delete it (for consistency if the parameters change)
    if os.path.exists(out_path + '.jpg'):
        os.remove(out_path + '.jpg')

    # driver
    options = webdriver.ChromeOptions()
    options.headless = True
    
    # not download files automatically
    #prefs = {"download.prompt_for_download": True}
    #options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome('/home/lugeon/drivers/chromedriver', options=options)
    
    driver.set_page_load_timeout(10) #10 seconds timeout
    
    must_convert = False

    # access the url, takes a screenshot (only in png)
    try:
        driver.get(url)
        
        # with given dimensions
        driver.set_window_size(in_width, in_height) # May need manual adjustment
        driver.find_element_by_tag_name('body').screenshot(out_path + '.png')
        
        # only upper part
        #driver.save_screenshot(out_path + '.png')
        
        must_convert = True

    except:
        driver.quit()
        return

    # if screenshot succeeds, convert the png into a jpeg of lesser dimensions and quality
    if must_convert:
        img = Image.open(out_path + '.png')
        img = img.convert('RGB')
        img = img.resize((out_width, out_height), Image.ANTIALIAS)
        img.save(out_path + '.jpeg', optimize=True, quality=quality)
        os.remove(out_path + '.png')

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
