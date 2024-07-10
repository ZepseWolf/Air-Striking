import esprima
import fasttext
import numpy as np
import tensorflow as tf
import keras
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import ast  
from sklearn.metrics.pairwise import cosine_similarity
from python_module.websiteModule import get_javascript , get_domain_from_url , js_to_trainable_string_arr

num_samples = 1000
vocab_size = 20000
max_sequence_length = 20
embedding_dim = 50
len_of_phrases = 200
embedding_model = fasttext.load_model('code_embedding_model.bin')
test_string = 'type Program sourceType script body type ExpressionStatement expression type UnaryExpression prefix True operator ! argument type CallExpression callee type FunctionExpression expression False isAsync False id None params type Identifier name e body type BlockStatement body type IfStatement test type LogicalExpression operator && left type BinaryExpression operator == left type Literal value object raw "object" right type UnaryExpression prefix True operator typeof argument type Identifier name exports right type BinaryExpression operator != left type Literal value undefined raw "undefined" right type UnaryExpression prefix True operator typeof argument type Identifier name module consequent type ExpressionStatement expression type AssignmentExpression operator = left type MemberExpression computed False object type Identifier name module property type Identifier name exports right type CallExpression callee type Identifier name e arguments alternate type IfStatement test type LogicalExpression operator && left type BinaryExpression operator == left type Literal value function raw "function" right type UnaryExpression prefix True operator typeof argument type Identifier name define right type MemberExpression computed False object type Identifier name define property type Identifier name amd consequent type ExpressionStatement expression type CallExpression callee type Identifier name define arguments type ArrayExpression elements type Identifier name e alternate type BlockStatement body type VariableDeclaration declarations type VariableDeclarator id type Identifier name t init None kind var type ExpressionStatement expression type SequenceExpression expressions type AssignmentExpression operator = left type Identifier name t right type ConditionalExpression test type BinaryExpression operator != left type Literal value undefined raw "undefined" right type UnaryExpression prefix True operator typeof argument type'


try:
    phrase = test_string.replace("\n"," ")
    print(embedding_model.get_sentence_vector(phrase))
except Exception as error:
    print("An exception occurred" , str(error)[:400])