import pandas as pd

reader = pd.read_json("/dlabdata1/lugeon/websites_40000_5cat_html.json.gz", orient='records', lines=True, chunksize=1000)
df = pd.DataFrame([])
for chunk in reader:
    df = pd.concat((df, pd.DataFrame(chunk)))