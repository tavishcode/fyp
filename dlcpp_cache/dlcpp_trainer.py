from keras.layers import Input, Dense
from keras.models import Sequential
from keras import losses
from keras.utils import plot_model
from keras.models import load_model
from math import log10
from collections import defaultdict
import numpy as np
import sys

class DlcppTrainer:
    def __init__(self, num_of_contents, training=True):
        self.name = "dlcpp"
        self.NUM_CONTENT_TYPES = num_of_contents
        self.model = False
        self.pop_levels = 20
        self.current_ranking = defaultdict(int)
        self.TRAINING = training
        if not self.TRAINING:
            self.model = self.load_trained_model('./dlcpp_cache/dlcpp_model.h5')


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
        for content_type in range(self.NUM_CONTENT_TYPES):
            input_features.append([0, num_requests, content_sum, request_entropy])

        for content_type in req_hist.keys():
            ix = int(content_type[7:])
            input_features[ix][0] = req_hist[content_type]
        
        return np.array(input_features)

    def get_true_labels(self,req_hist):

        # if popularity is first = [1,0,0,0,0, ..,0,0]
        # if popularity is 0.43 -> (1 - 0.43) -> 0.57*100 -> 57% / pop_levels(10) -> int(5.7) -> 5 -> [0,0,0,0,0,1,0,0,0,0]
        true_labels = []
        for content_type in range(self.NUM_CONTENT_TYPES):
            true_labels.append(np.zeros(self.pop_levels))
            true_labels[-1][-1] = 1

        num_requests = sum(req_hist.values())

        for content_type in req_hist.keys():
            ix = int(content_type[7:])
            if req_hist[content_type] == 0:
                popularity = self.pop_levels-1
            else:
                popularity = int((100*(1 - (req_hist[content_type] / num_requests)) / self.pop_levels)) # 1st = 0, last = 9
            true_labels[ix][-1] = 0
            true_labels[ix][popularity] = 1

        true_labels = np.array(true_labels)
        return true_labels

    def train(self,model, prev_req_hist, req_hist):
        if not self.model:
            self.model = baseline_model()
        else:
            input_features = extract_features(prev_req_hist)
            true_labels = get_true_labels(req_hist)
            print(input_features)
            print(true_labels)
            self.model.fit(input_features, true_labels, epochs=10, verbose=2)

    def get_entropy_csv(self,req_prob):
        request_entropy_array = []
        for content_probability in req_prob:
            request_entropy_array.append(content_probability * log10(content_probability) if content_probability != 0 else 0)
        request_entropy = (-1) * sum(request_entropy_array)
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
            if content_probability == 0.00000:
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
            true_labels = self.get_true_labels_csv(curr_reqs,reqs_per_row)
            # print(true_labels)
            self.model.fit(input_features, true_labels, epochs=100, verbose=0)
            # score = self.model.evaluate(input_features,true_labels)
            # print(score)

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
        return np.argmax(prediction,axis = 1)

    def updated_popularity(self,curr_reqs):
        ranks = self.predict(curr_reqs)
        for ix in range(self.NUM_CONTENT_TYPES):
            self.current_ranking['content'+str(ix)] = ranks[ix]
        return self.current_ranking

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(5, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Dense(self.pop_levels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_trained_model(self,file):
        return load_model(file)


    def report(self):
        print(self.model.summary())
        self.model.save('./dlcpp_cache/dlcpp_model.h5')
