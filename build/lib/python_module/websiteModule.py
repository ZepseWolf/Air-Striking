import requests
import esprima
from urllib.parse import urlparse

def script_to_dict(script_node):
    if hasattr(script_node, '__dict__'):
        node_dict = {}
        for key, value in script_node.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, list):
                node_dict[key] = [script_to_dict(item) for item in value]
            else:
                node_dict[key] = script_to_dict(value)
        return node_dict
    elif isinstance(script_node, (str, int, float)):
        return script_node
    else:
        return str(script_node)
    
def flatten_json(json_obj):
    flattened_list = []

    def traverse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                flattened_list.append(key)
                if isinstance(value, (str, int, float)):
                    flattened_list.append(str(value))
                else:
                    traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (str, int, float)):
                    flattened_list.append(str(item))
                else:
                    traverse(item)

    traverse(json_obj)
    return flattened_list

def js_to_trainable_string_arr(eg: str):
    
    try:
        return flatten_json(script_to_dict(esprima.parseScript(eg)))
    except Exception as error:
        print("error : " , error)
        return []

def get_domain_from_url(url: str):
    return urlparse(url).netloc

def get_javascript(url:str):

    checkIsJs =  url[len(url)-3:] == ".js" or (url.find('?') and url[url.find('?')-3:url.find('?')] == ".js" )
    if checkIsJs :
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content
                return response.text
            else:
                print("Failed to retrieve data from URL:", response.status_code)
                return ""
        except Exception as e:
            print("An error occurred:", e)
            return ""
    else:
        print("Not JS link")
        return ""