
from keras.layers import Input, Dense
from keras.models import Sequential
from keras import losses
from math import log10
from router import Router
import numpy as np
import sys
sys.path.insert(0, './src')
from contentstore import DlcppContentStore


def predict_popularity(model,contentstore,timestamp):
    #Training data from 2 time delta back
    input_features = []
    req_hist_prev = contentstore.req_hist_prev # dict of [Content Type] : number of hits
    print("req_hist_prev",req_hist_prev)
    num_requests = sum(req_hist_prev.values()) 
    content_sum = len(req_hist_prev)
    request_entropy_array = []

    for content_type in req_hist_prev.keys():
        content_request = req_hist_prev[content_type]
        try:
            content_probabity = content_request/num_requests
        except:
            content_probabity = 0
        request_entropy_array.append(content_probabity * log10(content_probabity) if content_probabity != 0 else 0)
        
    request_entropy = (-1) * sum(request_entropy_array)
    # print("request_entropy_array",request_entropy_array)
    # print("request_entropy",request_entropy)

    for content_type in req_hist_prev:
        content_request = req_hist_prev[content_type]
        input_features.append([num_requests,content_request,content_sum,request_entropy])

    input_features = np.array(input_features)
    print("input features",input_features)

    #Calculate actual popularity **levels**(1 to 10) from previous time delta
    req_hist = contentstore.req_hist
    print("req_hist",req_hist)
    req_hist_level = []
    pop_levels = 10
    num_requests_correct = sum(req_hist.values())
    for content_type in req_hist.keys():
        if req_hist[content_type] == 0:
            req_hist_level.append(pop_levels-1)
        else:
            req_hist_level.append(int((100*(1 - (req_hist[content_type] / num_requests_correct)) / pop_levels)) )#1st = 0, last = 9

    req_hist_level_vector = np.zeros((len(req_hist_level),pop_levels), dtype=int) #2D array of [Content type] [pop_levels]
    req_hist_level_vector[np.arange(len(req_hist_level)), req_hist_level] = 1 
    #if popularity is first = [1,0,0,0,0, ..,0,0]
    #if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
    print("req_hist_level_vector",req_hist_level_vector)
    if not input_features.size == 0:
        model.fit(input_features,req_hist_level_vector, epochs=10,verbose=2)
 

def baseline_model(n):
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(Dense(10, input_dim=8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

            
        
            
        

        
            


