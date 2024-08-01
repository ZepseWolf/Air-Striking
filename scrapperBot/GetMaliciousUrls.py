import pandas as pd

df = pd.read_parquet('./database/train.parquet')

json_data = []
json_url =[]
countPhishing=0
countLegit=0


print("Phishing Links:")
for _, row in df.iloc[3099:3300].iterrows():
    df_url = row['url']
    # Check if the URL exists in the JSON data
  
    if row['status'] == "phishing":
        print(df_url )


