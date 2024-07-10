import pandas as pd
import requests
import re
import html
import ssl
import socket
import http.client
import bs4
import json
from python_module.websiteModule import get_javascript , get_domain_from_url , js_to_trainable_string_arr

def get_all_urls(js_code):
    url_pattern = re.compile(r'<script.*?src=["\'](https?://[^\s"\']+).*?<\/script>', re.DOTALL)
    matches = re.findall(url_pattern, js_code)

    return matches

def extract_main_domain(url):
    url = url.split("://")[-1].split("/")[0].lstrip("www.")
    return url

def get_peer_certificate(host, port=443):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
            s.connect((host, 443))
            cert = s.getpeercert()
            certificate_information = {
                'C': cert['issuer'][0][0][1],
                'O': cert['issuer'][1][0][1],
                'CN': cert['issuer'][2][0][1],
                'notBefore': cert['notBefore'],
                'notAfter': cert['notAfter'],
            }
            print(cert)
            return certificate_information
            
    except ssl.SSLError as e:
        print(f"SSL Error: {e}")
    except socket.error as e:
        print(f"Socket Error: {e}")
    except http.client.HTTPException as e:
        print(f"HTTP Exception: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None

def remove_non_alphanumeric_starting_elements(arr):
    # Regular expression pattern to match elements that don't start with an alphabet or a number
    pattern = re.compile(r'^[^a-zA-Z0-9]+')
    filtered_arr = [item for item in arr if not pattern.match(item)]
    return filtered_arr

def get_content(text):
    # Regular expression pattern to match URLs (with or without http/https prefix)
    text = html.unescape(text)
    # Get all script src path 
    soup = bs4.BeautifulSoup(text, features='html.parser')
    scripts = soup.find_all('script')
    srcs = [link['src'] for link in scripts if 'src' in link.attrs]
    text = text.replace('\n', '').replace('\t', '')
    # url_pattern = r'https?://\S+?(?=["\'])'
    pattern = r'<script\b[^>]*>.*?</script>'
    # Remove all <script> tags and their content from the HTML
    modified_html1 = re.sub(pattern, '', text, flags=re.DOTALL)
    pattern1 = r'<style\b[^>]*>.*?</style>'
    # Remove all <script> tags and their content from the HTML
    modified_html = re.sub(pattern1, '', modified_html1, flags=re.DOTALL)
    url_pattern = r'<[^>]*?>([^<]+)</[^>]*?>'
    # Find all matches using the regex pattern
    urls =  re.findall(url_pattern, modified_html)
    trimmed_urls = [url.strip() for url in urls]
    
    #joint_string = trimmed_urls #+ remove_non_alphanumeric_starting_elements(get_all_data_from_script(text)) 
    
    return ' '.join(trimmed_urls), srcs 

def get_HTML_and_cert(url):
    try:
        print("At " , url)
        response = requests.get(url)
        if response.status_code == 200:
            return response.text , get_peer_certificate(extract_main_domain(url))  # HTML content of the webpage
        else:
            print(f"Failed to retrieve the content. Status code: {response.status_code}")
            return None, None
    except requests.RequestException as e:
        print("Error:", e)
        return None, None
    
#Provide the file path to your Parquet file
file_path = './database/train.parquet'

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")

        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:

            return output
        return K.sum(output, axis=1)

# Read the Parquet file
df = pd.read_parquet(file_path)
json_data = []
phishingCount = 0
legitCount =0
# for index, row in df.iterrows():
for index, row in df.iloc[160:].iterrows(): 
    index = index

    url = row['url']
    htmlString , cert= get_HTML_and_cert(url)
    

    if htmlString is not None:
        if row['status'] == "legitimate": # "legitimate"/0 or "phishing"/1
            classification = 0
        elif row['status'] == "phishing":
            classification = 1

        signatures, websites= get_content(htmlString)

        # Clean URL and JS 
        src_combined =[]
        for src in websites:
            targetUrl = src
            if src[:2] == "//":
                targetUrl = src[2:]
        
            elif src[:1] == "/" :
                targetUrl = get_domain_from_url(url)+src
            else:
                targetUrl = src
            src_combined += js_to_trainable_string_arr(get_javascript(src))

        if signatures != "" or len(websites) != 0  :
            # Ensure not all empty
            json_data.append({
                "url" : url,
                "certs" :  cert,
                "signatures" :signatures,
                "websites" :src_combined,
                "classification": classification,
                "index": index
            })

    if index%20 == 0 and index >0:
        n = int(index/20)
        
        with open("./database/signatures/websites"+str(n)+".json", 'w') as json_file:
            json.dump(json_data, json_file, indent=4) 
        json_data = []
    
    if index >6000:
        break