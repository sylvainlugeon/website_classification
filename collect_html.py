import numpy as np
import pandas as pd

import requests
import multiprocessing as mp
import json
import gzip


def main():
    """"Entry point"""

    folder = '/dlabdata1/lugeon/'
    #folder = '../data/'
    name = "websites_40000_5cat"
    ext = '.gz'
    step = 100
    timeout = 10
    feedback = 1000

    data = pd.read_csv(folder + name + ext, header=0, names=['id', 'url', 'cat0'])
    data = data[['url', 'cat0']]
    
    print('retrieving html pages from', name)
    
    write_html_to_file(df=data, step=step, path=folder+name+'_html.json.gz', timeout=timeout, feedback=feedback)


def get_homepage(url, timeout):
    """Return the html page corresponding to the url, or None if there was a request error"""

    try:
        return requests.get(url, timeout=timeout).text
    except(Exception):
        return None



def worker(start_id, end_id, df, q, timeout):
    """A worker that retrieve the html pages for a given slice of a dataframe and put them on the queue"""

    # in case the df length is not divisible by the step size
    if end_id > df.shape[0]:
        end_id = df.shape[0]

    # copying the df and retrieving the html pages
    sliced_df = df.iloc[start_id: end_id].copy()
    sliced_df['html'] = sliced_df.apply(lambda row: get_homepage(row.url, timeout), axis=1)

    # putting in the queue
    sliced_df.apply(lambda row: q.put({
        'url': row.url,
        'html': row.html,
        'cat': row.cat0
    }), axis=1)

    return



def listener(q, path, nb_lines, feedback):
    """A listener that takes elements in the queue and write them into a file, in json format"""

    i = 1 # to control the format

    with gzip.open(path, 'wt') as f:

        print('listening...')
        #f.write('[')
        f.flush()

        # listening on the queue
        while 1:
            m = q.get()

           # if all the workers are done
            if m == 'kill':
                #f.write(']')
                f.flush()
                print('done writing to file')
                break

            # else write the worker output on the file
            f.write(json.dumps(m))

            # for the format
            if(i < nb_lines):
                f.write('\n')
                i += 1

            f.flush()

            # verbose
            if (i % feedback == 0):
                print('{}/{} written to file'.format(i, nb_lines))



def write_html_to_file(df, step, path, timeout, feedback):
    """Retrieve and write in a file the html pages from urls in a dataframe"""

    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool(mp.cpu_count())

    # put listener to work first, will occupy 1 thread
    watcher = p.apply_async(listener, (q, path, df.shape[0], feedback))

    # fire off workers
    jobs = []

    # for each slice of the dataframe, a thread retreive the html pages
    for i in range(0, df.shape[0], step):
        job = p.apply_async(worker, (i, i+step, df, q, timeout))
        jobs.append(job)

    # to be sure that all jobs are done
    for job in jobs:
        job.get()

    # kill the listener
    q.put('kill')
    p.close()
    p.join()



if __name__ == "__main__":
    main()
