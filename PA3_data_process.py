#!/usr/bin/env python
# coding: utf-8

# # Data processing for image captioning

# In[1]:


import numpy as np
import re
from pickle import load,dump
from nltk import FreqDist
from numpy import array
from text import Tokenizer


# In[2]:


# make dictionary {"image_name":[caption_list]}
descriptions = dict()

with open("data/Flickr8k/Flickr8k.token.txt") as f:
    data = f.read()

# try:
for el in data.split("\n"):
    tokens = el.split()
    if len(tokens) < 2:
        continue
    image_id , image_desc = tokens[0],tokens[1:]

    # dropping .jpg from image id
    image_id = image_id.split(".")[0]

    image_desc = " ".join(image_desc)
    
    # check if image_id is already present or not
    if image_id in descriptions:
        descriptions[image_id].append(image_desc)
    else:
        descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
            
# except Exception as e: 
#     print("Exception got :- \n",e)


# In[3]:


descriptions["1000268201_693b08cb0e"]


# In[4]:


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[5]:


for k in descriptions.keys():
    value = descriptions[k]
    caption_list = []
    for ec in value:
        
        # replaces specific and general phrases
        sent = decontracted(ec)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        
        # startseq is for kick starting the partial sequence generation and endseq is to stop while predicting.
        # for more referance please check https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
        image_cap = 'startseq ' + sent.lower() + ' endseq'
        caption_list.append(image_cap)
    descriptions[k] = caption_list


# In[6]:


descriptions["1000268201_693b08cb0e"]


# In[7]:


# save dictionary
dump(descriptions,open("descriptions.pkl","wb"))


# In[8]:


# load the saved dictionary
with open("descriptions.pkl","rb") as f:
    descriptions = load(f)


# In[9]:


descriptions["2394267183_735d2dc868"]


# In[10]:


# train descriptions dictionary creation
train_descriptions = dict()
with open("data/Flickr8k/Flickr_8k.trainImages.txt","r") as f:
    data = f.read()
    
try:
    for el in data.split("\n"):
        tokens = el.split(".")
        image_id = tokens[0]
        if image_id in descriptions:
            train_descriptions[image_id] = descriptions[image_id]
                    
except Exception as e:
    print("Exception got :- \n",e)


# In[11]:


#save the file
dump(train_descriptions,open("train_descriptions.pkl","wb"))


# In[12]:


# creating corpus
corpus = ""

with open("train_descriptions.pkl","rb") as f:
    train_descriptions = load(f)
for ec in train_descriptions.values():
    for el in ec:
        corpus += " "+el
total_words = corpus.split()
vocabulary = set(total_words)
print("The size of vocablury is {}".format(len(vocabulary)))


# In[13]:


# creating frequecny distribution of words
freq_dist = FreqDist(total_words)
freq_dist.most_common(5)


# In[14]:


#removing least common words from vocabulary
for ew in list(vocabulary):
    if(freq_dist[ew]<10):
        vocabulary.remove(ew)
VOCAB_SIZE = len(vocabulary)+1
print("Total unique words after remooving less frequent word from our corpus = {}".format(VOCAB_SIZE))


# In[15]:


train_descriptions['2513260012_03d33305cf']


# In[16]:


caption_list = []
for el in train_descriptions.values():
    for ec in el:
        caption_list.append(ec)
print("The total caption present = {}".format(len(caption_list)))


# In[17]:


token = Tokenizer(num_words=VOCAB_SIZE)
token.fit_on_texts(caption_list)


# In[18]:


# index to words are assigned according to frequency. i.e the most frequent word has index of 1
ix_to_word = token.index_word


# In[19]:


for k in list(ix_to_word):
    if k>=1665:
        ix_to_word.pop(k, None)
word_to_ix = dict()
for k,v in ix_to_word.items():
    word_to_ix[v] = k
print(len(word_to_ix))
print(len(ix_to_word))


# In[20]:


# finding the max_length caption
MAX_LENGTH = 0
temp = 0
for ec in caption_list:
    temp = len(ec.split())
    if(MAX_LENGTH<=temp):
        MAX_LENGTH = temp
print("Maximum caption has length of {}".format(MAX_LENGTH))


# In[21]:


with open('data/glove6B/glove.6B.300d.txt', 'rb') as f:
    data=f.read()
glove = dict()

try:
    for el in data.decode().split("\n"):
        tokens = el.split()
        if len(tokens) < 2:
            continue
        
        word = tokens[0]
        vec = []
        for i in range(1,len(tokens)):
            vec.append(float(tokens[i]))
        glove[word] = vec
        
except Exception as e: 
    print("Exception got :- \n",e)


# In[22]:


print(len(glove['the'])) # vector size
print(len(glove.keys())) # number of words


# In[23]:


EMBEDDING_SIZE = 300

# Geti 300-dim dense vector for each of the words in vocabulary
embedding_matrix = np.zeros((VOCAB_SIZE,EMBEDDING_SIZE))
embedding_matrix.shape


# In[24]:


EMBEDDING_SIZE = 300

# Get 300-dim dense vector for each of the words in vocabulary
embedding_matrix = np.zeros(((VOCAB_SIZE),EMBEDDING_SIZE))

for word, i in word_to_ix.items():
    embedding_vector = np.zeros(EMBEDDING_SIZE)
    if word in glove.keys():
        embedding_vector = glove[word]
        embedding_matrix[i] = embedding_vector
    else:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# save the embedding matrix to file
with open("embedding_matrix.pkl","wb") as f:
    dump(embedding_matrix,f)


# In[25]:


word_to_ix["the"]
#embedding_matrix[5] == glove['the'] : True


# In[26]:


word_to_ix


# In[27]:


embedding_matrix

