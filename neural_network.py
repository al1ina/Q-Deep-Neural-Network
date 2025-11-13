import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self,n_in,n_hidden,n_out,l_rate=0.001, gamma = 0.99):
        self.w1 = np.random.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.w2 = np.random.randn(n_hidden, n_out) / np.sqrt(n_hidden)
        # self.w1 = 0.01 * np.random.randn(n_in, n_hidden) # bc every hidden neural needs to acces every input one 
        #so every single column (hidden neuron) needs to access every single row (input neuron)
        self.b1 = np.zeros((1,n_hidden)) # bias for hidden layer
        # self.w2 = 0.01 * np.random.randn(n_hidden, n_out)
        self.b2 = np.zeros((1,n_out)) 

        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)
        
        # Adam variables
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)

        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

        self.t = 0 # initially set to 0, tells you which step you are on
        self.v_dw1 = np.zeros_like(self.w1)
        self.s_dw1 = np.zeros_like(self.w1)
        self.v_db1 = np.zeros_like(self.b1)
        self.s_db1 = np.zeros_like(self.b1)

        self.v_dw2 = np.zeros_like(self.w2)
        self.s_dw2 = np.zeros_like(self.w2)
        self.v_db2 = np.zeros_like(self.b2)
        self.s_db2 = np.zeros_like(self.b2)


    def forward(self,x): # x is the output of the previous layer
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.maximum(0,self.z1) # relu activation function
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2, (x, self.z1, self.a1, self.z2)  # output layer (no activation function for now)
    def predict(self,x): # where x is the state
        z2, _ = self.forward(x)
        return z2
    def zero_grad(self):
        self.dw1.fill(0)
        self.db1.fill(0)
        self.dw2.fill(0)
        self.db2.fill(0)
    def apply_gradients(self, l_rate = 0.001):
        self.w1 -= l_rate * self.dw1
        self.b1 -= l_rate * self.db1
        self.w2 -= l_rate * self.dw2
        self.b2 -= l_rate * self.db2
    def copy_from(self, other):
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
    def activation(z): # sigmoid, for now later change it to ReLU
        return 1 / (1 + np.exp(-z))
    def activation_derivative(self, z):
        s = self.activation(z)
        return s * (1 - s)
    def backward(self, x, y_true, batch_size, l_rate=0.001, beta1=0.9, beta2 = 0.999, epsilon= 10 ** (-8) ):
        self.t += 1  # increment teh timestep
        # Output Layer
        nabla_2 = (self.z2 - y_true) / batch_size # this is error and it creates an array
        # row is the batch size
        # columns are the number of output neurons
        delta_w2 = np.dot(self.a1.T, nabla_2)
        #takes the sum of each column "sum down the rows" think of sum of all contributions
        delta_b2 = np.sum(nabla_2, axis=0, keepdims=True)

        # Hidden Layer
        delta_a1 = np.dot(nabla_2, self.w2.T)
        nabla_1 = delta_a1 * (self.z1 > 0) # multiplied by the ReLU derivative
        delta_w1 = np.dot(x.T, nabla_1)
        delta_b1 = np.sum(nabla_1, axis=0, keepdims=True)

        # Adam Optimization
        self.v_dw1 = beta1 * self.v_dw1 + (1 - beta1) * delta_w1
        self.s_dw1 = beta2 * self.s_dw1 + (1 - beta2) * (delta_w1 ** 2)
        self.v_db1 = beta1 * self.v_db1 + (1 - beta1) * delta_b1
        self.s_db1 = beta2 * self.s_db1 + (1 - beta2) * (delta_b1 ** 2)

        self.v_dw2 = beta1 * self.v_dw2 + (1 - beta1) * delta_w2
        self.s_dw2 = beta2 * self.s_dw2 + (1 - beta2) * (delta_w2 ** 2)
        self.v_db2 = beta1 * self.v_db2 + (1 - beta1) * delta_b2
        self.s_db2 = beta2 * self.s_db2 + (1 - beta2) * (delta_b2 ** 2)

        # bias correction
        v_dw1_corr = self.v_dw1 / (1 - beta1 ** self.t)
        s_dw1_corr = self.s_dw1 / (1 - beta2 ** self.t)
        v_dw2_corr = self.v_dw2 / (1 - beta1 ** self.t)
        s_dw2_corr = self.s_dw2 / (1 - beta2 ** self.t)

        v_db1_corr = self.v_db1 / (1 - beta1 ** self.t)
        s_db1_corr = self.s_db1 / (1 - beta2 ** self.t)
        v_db2_corr = self.v_db2 / (1 - beta1 ** self.t)
        s_db2_corr = self.s_db2 / (1 - beta2 ** self.t)

        #Update
        self.w1 -= l_rate* v_dw1_corr / (np.sqrt(s_dw1_corr) + epsilon)
        self.w2 -= l_rate* v_dw2_corr / (np.sqrt(s_dw2_corr) + epsilon)
        self.b1 -= l_rate* v_db1_corr / (np.sqrt(s_db1_corr) + epsilon)
        self.b2 -= l_rate* v_db2_corr / (np.sqrt(s_db2_corr) + epsilon)

    
class ReplayBuffer:
    def __init__(self, capacity = 100):
        self.capacity = capacity
        self.storage = []
    def add(self, state, action, reward, next_state, done):
        self.storage.append((state, action, reward, next_state, done))
        if len(self.storage) > self.capacity:
            self.storage.pop(0)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.storage), batch_size, replace=False)
        random_array = []
        for x in indices:
            random_array.append(self.storage[x])
        return random_array
    
class TrainingLoop:
    def __init__(self, n_in, n_hidden, n_out, l_rate=0.01, gamma = 0.99, target_update_freq=100):
        self.nn = NeuralNetwork(n_in, n_hidden, n_out, l_rate, gamma)
        self.target_nn = NeuralNetwork(n_in, n_hidden, n_out, l_rate, gamma)  # target network
        self.target_nn.copy_from(self.nn)  # initialize target as a clone
        self.replay = ReplayBuffer(10000)
        
        self.gamma = gamma
        self.l_rate = l_rate
        self.target_update_freq = target_update_freq
        self.step_count = 0
    def train_step(self, batch_size):
        if len(self.replay.storage) < batch_size:
            return
        minibatch = self.replay.sample(batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            q_values = self.nn.predict(state)
            q_target = q_values.copy()

            if done:
                q_target[0, action] = reward
            else:
                q_next = self.target_nn.predict(next_state)
                q_target[0, action] = reward + self.gamma * np.max(q_next)
            states.append(state)
            targets.append(q_target)
            
        states = np.vstack(states)
        targets = np.vstack(targets)

        _ = self.nn.forward(states)
        # self.nn.forward(states)
        self.nn.backward(states, targets, states.shape[0], l_rate=self.l_rate)

        self.step_count += 1

        if self.step_count % self.target_update_freq == 0:
            self.target_nn.copy_from(self.nn)


        

