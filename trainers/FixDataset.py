import pandas as pd
import esprima
import json
from python_module.websiteModule import get_javascript , get_domain_from_url , js_to_trainable_string_arr


df = pd.read_parquet('./database/train.parquet')

json_data = []
json_url =[]
countPhishing=0
countLegit=0
for i in range(1,300):
    temp_json = None
    file_path = './database/signatures/websites'+str(i)+'.json'
    with open(file_path, 'r') as file:
        # Load JSON data from the file
        temp_json = json.load(file)
        
        for i , row in enumerate(temp_json):

            current_websites = row['websites']
            # temp_json[i]['orginalJS'] = js_to_trainable_string(row['orginalJS'])
            current_classification = row['classification']
            for ii , row in enumerate(current_websites):
                # esprima it then save the ast
                targetUrl = ""
                if temp_json[i]["websites"][ii][:2] == "//":
                    targetUrl = temp_json[i]["websites"][ii][2:]
         
                elif temp_json[i]["websites"][ii][:1] == "/" :
                    targetUrl = get_domain_from_url(temp_json[i]["url"])+temp_json[i]["websites"][ii]
                else:
                    targetUrl = temp_json[i]["websites"][ii]
                jsVal = get_javascript(targetUrl)

                if not jsVal is None:
                    temp_json[i]["websites"][ii] = jsVal
                else: 
                    del temp_json[i]["websites"][ii]
            
            if current_classification == 0:
                countLegit += 1
            elif current_classification == 1:
                countPhishing += 1
                
    if not temp_json is None:
        with open(file_path, 'w') as file:
            json.dump(temp_json, file)
    # Iterate through the DataFrame

        
# for _, row in df.iterrows():
    # df_url = row['url']
    # Check if the URL exists in the JSON data
    # for entry in json_data:
    #     if 'url' in entry and entry['url'] == df_url:
            # Update the JSON data as needed
 
            # entry['classification'] = row['status']
print("Legit : " , countLegit, " Phishing : " , countPhishing )
# with open("./database/signatures/websitesMain.json", 'w') as file:
#     json.dump(json_data, file, indent=2)

