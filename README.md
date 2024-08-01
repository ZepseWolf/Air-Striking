# Air Striking

A repository that aims to research methods to fight against scam website specifically phishing website

## Prerequisites that needs to run once
1. Install modules using `pip install -r requirements.txt`
2. Run `cd trainers`
3. Run `py PreTrainWordEmbedding.py` to use fastword unsupervised learning saved at code_embedding_model.bin 
4. Run `py FinalProcessing.py` to get a saved variable so you do not need to process again saved as refined_variables.pkl

## Start Scraping Training & Testing data
1. Go to [https://huggingface.co/pirocheto/phishing-url-detection] and download the test.parquet and train.parquet. Put inside *database* folder.
1. Run `cd scrapperBot`
2. Run `py scrapper.py`

## Start Server
1. Run `cd server`
2. Run `py server.py`

## Train dataset (ensure that data is scraped)
1. Run `cd trainers`
2. (Optional) Run `py PreTrainWordEmbedding.py` if file code_embedding_model.bin doesnt exist
3. (Optional) Run `py FinalProcessing.py`  if file refined_variables.pkl doesnt exist
4. Run `py CodeAiTraining.py` for code based 
5. Run `py TextAiTraining.py` for text based 


## Generate Analysis with confusion matrix 
1. Run `cd trainers/analytic`
2. Run `py SubsetAnalysis.py`  

## Install toolbar (ensure that server is turn on)
1. Open chrome , go to [chrome://extensions/] , turn on developer mode. 
2. Click on load unpacked , select folder [Air_Striking/frontend/].

## Local bypass for browser and local server interaction (can be dangerous to security)
1. Right click chrome application on desktop, click *property*, on *target* field input
`C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --user-data-dir --allowed-origins="http://localhost:5000`
