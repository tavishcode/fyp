
# from keras.layers import Input, Dense
# from keras.models import Model
from math import log2
from router import Router
import numpy as np


def predict_popularity(routers,timestamp):
    #Training data from 2 time delta back
    for router in routers:
        input_features = []
        req_hist = router.contentstore.req_hist # dict of [Content Type] : number of hits
        num_requests = sum(req_hist.values()) 
        content_sum = len(req_hist)
        request_entropy_array = []

        for content_type in req_hist.keys():
            content_request = req_hist[content_type]
            content_probabity = content_request/num_requests
            request_entropy_array.append((-1) *  content_probabity * log2(content_probabity))
            
        request_entropy = sum(request_entropy_array)

        for content_type in req_hist:
            input_features.append([request,content_request,content_sum,request_entropy])

        input_features = np.array(input_features)
        print(input_features)

        #Calculate actual popularity **levels**(1 to 10) from previous time delta
        correct_pop = router.contentstore.correct_pop
        correct_pop_level = []
        pop_levels = 10
        num_requests_correct = sum(correct_pop.values())
        for content_type in correct_pop.keys():
            correct_pop_level.append(int((100*(1 - (correct_pop[content_type] / num_request_correct)) / pop_levels)) )#1st = 0, last = 9
        
        correct_pop_level_vector = np.zeros((len(correct_pop_level),pop_levels), dtype=int) #2D array of [Content type] [pop_levels]
        correct_pop_level_vector[np.arange(len(correct_pop_level)), correct_pop_level] = 1 
        #if popularity is first = [1,0,0,0,0, ..,0,0]
        #if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
        

        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(10, input_dim=8, activation='softmax'))
        model.compile(loss='categorical_entropy', optimizer='adam', metrics=['accuracy'])


            
        
            
        

        
            


