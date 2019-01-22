# Reference: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

# Using DDPG for caching

def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])  
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        V = Dense(CONTENT_TYPES, activation='softmax')(h1)
        model = Model(input=S,output=V)
        print("We finished building the model")
        return model, model.trainable_weights, S

def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')    
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)  
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("We finished building the model")
        return model, A, S 

def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

 for j in range(max_steps):
        a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
        ob, r_t, done, info = env.step(a_t[0])

