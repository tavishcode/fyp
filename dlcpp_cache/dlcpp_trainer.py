from keras.layers import Input, Dense
from keras.models import Sequential
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from math import log10
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


# f = open('dlcpp-current-ranks.csv', 'w')
# w = csv.writer(f)
g = open('dlcpp-current-labels.csv', 'w')
p = csv.writer(g)

class DlcppTrainer:
    def __init__(self, num_of_contents, training=True):
        self.name = "dlcpp"
        self.NUM_CONTENT_TYPES = num_of_contents
        self.pop_levels = 10
        self.model = self.sigmoid_model()
        self.current_ranking = defaultdict(int)
        self.TRAINING = training
        if not self.TRAINING:
            self.model = self.load_trained_model('./dlcpp_cache/dlcpp_model.h5')
        print(K.tensorflow_backend._get_available_gpus())
        self.order = []


    def get_entropy(self,req_hist):
        num_requests = sum(req_hist.values()) 
        request_entropy_array = []
        for content_type in req_hist.keys():
            content_request = req_hist[content_type]
            try:
                content_probability = content_request/num_requests
            except:
                content_probability = 0
            request_entropy_array.append(content_probability * log10(content_probability) if content_probability != 0 else 0)
        request_entropy = (-1) * sum(request_entropy_array)
        return request_entropy

    def extract_features(self,req_hist):
        num_requests = sum(req_hist.values()) 
        content_sum = len(req_hist)
        request_entropy = self.get_entropy(req_hist)

        input_features = []
        for content_type in range(len(self.order)):
            ix = int(self.order[content_type][7:])
            input_features.append([0, num_requests, content_sum, request_entropy])
        for content_type in req_hist.keys():
            try:
                ix = self.order.index(content_type)
                input_features[ix][0] = req_hist[content_type]
            except:
                self.order.append(content_type)
                input_features.append([req_hist[content_type], num_requests, content_sum, request_entropy])

        return np.array(input_features)

    def get_true_labels(self,req_hist):
        # if popularity is first = [1,0,0,0,0, ..,0,0]
        # if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
        true_labels = []
        for content_type in range(len(self.order)):
            true_labels.append(np.zeros(self.pop_levels))
            true_labels[-1][-1] = 1

        num_requests = sum(req_hist.values())

        for content_type in req_hist.keys():
            try:
                ix = self.order.index(content_type)
                if req_hist[content_type] == 0:
                    popularity = self.pop_levels-1
                else:
                    popularity = int((100*(1 - (req_hist[content_type] / num_requests)) / self.pop_levels)) # 1st = 0, last = pop_levels-1
                true_labels[ix][-1] = 0
                true_labels[ix][popularity] = 1
            except:
                pass
           
        true_labels = np.array(true_labels)
        return true_labels

    def get_true_labels_ranks(self,req_hist):
        # req_hist_ranked = sorted(req_hist.items(),key = lambda x: x[1], reverse = True)
        # print(req_hist)
        true_labels=[]
        for content_type in self.order:
            i = 0
            try:
                ix = req_hist[content_type]
                # print(ix)
                if ix >= self.pop_levels:
                    ix = self.pop_levels - 1
                true_labels.append(np.zeros(self.pop_levels))
                true_labels[-1][ix] = 1
            except:
                true_labels.append(np.zeros(self.pop_levels))
                # true_labels[i][0] = 1
            i += 1
            # print("i",true_labels[i])

        true_labels = np.array(true_labels,dtype=int)
        return true_labels

    def get_true_labels_probability(self,req_hist):
        num_requests = sum(req_hist.values())
        true_labels=[]
        for content_type in self.order:
            i = 0
            try:
                reqs = req_hist[content_type]
                probability = reqs/num_requests
                true_labels.append(probability)
            except:
                print("8888888888888888888888888888888888")
                true_labels.append(0)
                # true_labels[i][0] = 1
            i += 1
            # print("i",true_labels[i])

        true_labels = np.array(true_labels,dtype=np.float32)
        return true_labels
        


    def train(self, prev_req_hist, req_hist):
        # if not self.model:
        #     self.model = self.baseline_model()
        # else:
        print("**********************")
        input_features = self.extract_features(prev_req_hist)
        true_labels = self.get_true_labels_probability(req_hist)
        # print(true_labels)
        print("order",len(self.order))
        print("input",input_features.shape)
        print(true_labels)
        p.writerow(true_labels)
        self.model.fit(input_features, true_labels, epochs=10, verbose=1)

    def get_entropy_csv(self,req_prob):
        request_entropy_array = []
        for content_probability in req_prob:
            request_entropy_array.append(content_probability *log10(content_probability))
        request_entropy = (-1) * sum(request_entropy_array)
        # print(request_entropy_array)
        # print(request_entropy)
        return request_entropy

    def extract_features_csv(self,req_prob, reqs_per_row):
        num_requests = reqs_per_row
        content_sum = len(req_prob)
        request_entropy = self.get_entropy_csv(req_prob)

        input_features = []
        i = 0
        for content_type in range(self.NUM_CONTENT_TYPES):
            input_features.append([int(req_prob[i]*reqs_per_row), num_requests, content_sum, request_entropy])
            i +=1
        return np.array(input_features)

    def get_true_labels_csv(self,req_prob, reqs_per_row):

        # if popularity is first = [1,0,0,0,0, ..,0,0]
        # if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
        true_labels = []
        # print(req_prob)
        for content_type in range(self.NUM_CONTENT_TYPES):
            true_labels.append(np.zeros(self.pop_levels))
            true_labels[-1][-1] = 1
        ix=0
        for content_probability in req_prob:
            if content_probability <= 0.01:
                popularity = self.pop_levels-1
            else:
                popularity = int((100*(1 - content_probability) / (100/self.pop_levels))) # 1st = 0, last = 9
            true_labels[ix][-1] = 0
            true_labels[ix][popularity] = 1
            ix += 1

        true_labels = np.array(true_labels)
        return true_labels

    def train_from_csv(self,prev_reqs, curr_reqs, reqs_per_row):
        if not self.model:
            self.model = self.baseline_model()
        else:
            input_features = self.extract_features_csv(prev_reqs,reqs_per_row)
            # print(input_features)
            true_labels = self.get_true_labels_csv(curr_reqs,reqs_per_row)
            # print(true_labels)
            self.model.fit(input_features, true_labels, epochs=50, verbose=0, validation_split=0.3)
            score = self.model.evaluate(input_features,true_labels)
            print(score)

    """TODO Simple input features"""
    def train_from_csv_2(self,prev_reqs, curr_reqs, reqs_per_row):
        if not self.model:
            self.model = self.baseline_model()
        else:
            entropy = self.get_entropy_csv(prev_reqs)
            print(prev_reqs)
            input_features = np.split(np.append(prev_reqs,entropy),self.NUM_CONTENT_TYPES+1)
            print(input_features)
            true_labels = self.get_true_labels_csv(curr_reqs,reqs_per_row)
            # print(true_labels)
            self.model.fit(input_features, true_labels, epochs=100, verbose=0)
            score = self.model.evaluate(input_features,true_labels)
            print(score)

    def evaluate_csv(self,curr_reqs,prev_reqs,reqs_per_row):
        X = self.extract_features_csv(prev_reqs,reqs_per_row)
        y = self.get_true_labels_csv(curr_reqs,reqs_per_row)
        return self.model.evaluate(X,y)

    def evaluate(self,curr_reqs,prev_reqs):
        X = self.extract_features(prev_reqs)
        y = self.get_true_labels(curr_reqs)
        return self.model.evaluate(X,y)
    

    def predict(self, data):
        input_features = self.extract_features(data)
        # print(input_features)
        prediction = self.model.predict(input_features)
        # print(prediction)
        # return np.argmax(prediction,axis = 1)
        return prediction

    def updated_popularity(self,curr_reqs):
        # ranks = self.predict(curr_reqs)
        preds= self.predict(curr_reqs)

        
        # print("Ranks",preds)
        for ix in range(len(self.order)):
            self.current_ranking[self.order[ix]] = preds[ix]

        self.current_ranking = OrderedDict(sorted(self.current_ranking.items(),key = lambda x: x[1], reverse=True))
        with open('current_ranking.csv', 'w') as f:
            for key in self.current_ranking.keys():
                f.write("%s,%s\n"%(key,self.current_ranking[key]))

        return self.current_ranking.keys()

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(4, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Dense(self.pop_levels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def sigmoid_model(self):
        model = Sequential()
        model.add(Dense(5, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def load_trained_model(self,file):
        return load_model(file)


    def report(self):
        print(self.model.summary())
        self.model.save('./dlcpp_cache/dlcpp_model.h5')

    def plot_metrics(self,loss,accuracy):
        plt.plot(accuracy,np.arange(1,accuracy.len))
        plt.ylabel('accuracy')
        plt.xlabel('batch')
        plt.show()
        plt.plot(loss,np.arange(1,loss.len))
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.show()
