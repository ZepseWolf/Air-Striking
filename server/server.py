from flask import Flask, url_for, jsonify, request, redirect
from flask_cors import CORS
from python_module.websiteModule import get_javascript, get_domain_from_url, js_to_trainable_string_arr

import pandas as pd
import requests
import re
import html
import ssl
import socket
import http.client
import bs4

import fasttext
import numpy as np
import tensorflow as tf

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger

import keras
from keras import backend as K
from keras.utils import custom_object_scope
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from attention import attention

import pickle

num_samples = 1000
size_of_vocabulary = 20000
max_sequence_length = 20
embedding_dim = 50
max_len = 200

def get_all_urls(js_code):
    url_pattern = re.compile(
        r'<script.*?src=["\'](https?://[^\s"\']+).*?<\/script>', re.DOTALL)
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
            # print(cert)
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
    urls = re.findall(url_pattern, modified_html)
    trimmed_urls = [url.strip() for url in urls]

    # joint_string = trimmed_urls #+ remove_non_alphanumeric_starting_elements(get_all_data_from_script(text))

    return ' '.join(trimmed_urls), srcs

def get_HTML_and_cert(url):
    try:
        print("At ", url)
        response = requests.get(url)
        if response.status_code == 200:
            # HTML content of the webpage
            return response.text, get_peer_certificate(extract_main_domain(url))
        else:
            print(
                f"Failed to retrieve the content. Status code: {response.status_code}")
            return None, None
    except requests.RequestException as e:
        print("Error:", e)
        return None, None

try:
    with open('tokenizer.pkl', 'rb') as f:
        loaded_tokenizer = pickle.load(f)
except FileNotFoundError:
    print("Tokenizer file not found. Please check the file path.")
    exit()  # Exit the script or handle the error accordingly

text_cnn_model = tf.keras.models.load_model(
    'text_classification_cnn.h5')  # Load your saved model
text_biLSTM_model = tf.keras.models.load_model(
    'text_classification_biLSTM_attention.h5', custom_objects = {'attention': attention})  # Load your saved model
code_cnn_model = tf.keras.models.load_model(
    'code_classification_cnn3.h5')  # Load your saved model
code_biLSTM_model = tf.keras.models.load_model(
    'code_classification_biLSTM_attention3.h5', custom_objects = {'attention': attention})  # Load your saved model

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# @app.route('/api/checkWebsite', methods=['GET'])
# def get_data():
#     return jsonify({
#           "id": "1234",
#           "cat": 202,
#           "website": ["tet", "erm"]
# 	})


@app.route('/api/checkWebsite', methods=['POST'])
def add_data():
    new_data = request.json

    if new_data:
        url = new_data["target_url"]
        domain = get_domain_from_url(new_data["target_url"])
        htmlString, cert = get_HTML_and_cert(url)
        
        if htmlString is not None:
            signatures, websites= get_content(htmlString)

            # Clean URL and JS 
            # src_combined =[]
            # for src in websites:
            #     targetUrl = src
            #     if src[:2] == "//":
            #         targetUrl = src[2:]
            
            #     elif src[:1] == "/" :
            #         targetUrl = get_domain_from_url(url)+src
            #     else:
            #         targetUrl = src
            #     src_combined += js_to_trainable_string_arr(get_javascript(src))

            if signatures != "":
                t = loaded_tokenizer.texts_to_sequences([signatures])
                tokenized_text = tf.keras.utils.pad_sequences(t, max_len, padding='post', value=0)
                y_pred_prob = text_cnn_model.predict(tokenized_text)
                print("Current url scored : " , y_pred_prob[0][0] )
                return jsonify({'isAffected': False if round(y_pred_prob[0][0], 0) == 0 else True ,'message': 'Data added successfully.'}), 201
            else:
                # signature not found
                pass
            # if len(src_combined) != 0  :
                # if there is website
            #     pass
            # else:
            #     pass
                
            return jsonify({'isAffected': True ,'message': 'Data added successfully.'}), 201
    else:
        return jsonify({'error': 'No data provided'}), 400

@app.errorhandler(404)
def page_not_found(e):
    return redirect("/", code=404)

@app.route("/")
def all_path():
	return f"Note that this website has no front facing interface."


if __name__ == "__main__":
	app.run()