import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from PIL import Image
import os

import pandas as pd
import numpy as np


def main():
    
    folder = "/dlabdata1/lugeon/"
    name = "websites_1000_5cat"
    ext = ".gz"

    df = pd.read_csv(folder + name + ext, names = ['uid', 'url', 'cat0'], header=0)
    
    df = df.iloc[:10]

    in_width = 1920
    in_height = 2 * 1080

    down_factor = 2

    out_width = int(in_width / down_factor)
    out_height = int(in_height / down_factor)

    quality = 0.85
    
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome('/home/lugeon/drivers/chromedriver', options=options)
    
    def f(url, name):
        return take_screenshot(driver=driver, 
                               url=url, 
                               in_width=in_width,
                               in_height=in_height, 
                               name=name, 
                               out_width=out_width, 
                               out_height=out_height, 
                               quality=quality)
    
    out_folder = folder + name + "_screenshots/"
    
    df.apply(lambda row: f(url=row.url, name = out_folder + str(row.uid)), axis=1)

    driver.quit()

def take_screenshot(driver, url, in_width, in_height, name, out_width, out_height, quality):
    
    try:
        driver.get(url)
        driver.set_window_size(in_width, in_height) # May need manual adjustment   
        driver.find_element_by_tag_name('body').screenshot(name + '.png')
        
    except:
        return
    
    img = Image.open(name + '.png')
    img = img.convert('RGB')
    img = img.resize((out_width, out_height), Image.ANTIALIAS)
    img.save(name + '.jpeg', optimize=True,quality=85)
    os.remove(name + '.png')
    
if __name__ == '__main__':
    main()