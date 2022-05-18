

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import copy

#%% 
filename = "G:/"
filename_2 = "1/"

filename_2_test = "2/"

user_num_extract = 10000
#date_extract_start = '2021-02-01 00:00:00'
#date_extract_end = '2021-03-15 00:00:00'

#%% Data reading
# user_id、sku_id、action_sequence、sku_name、first_Cate_id、first_cate_name
user_action_sequence_product_num = pd.read_csv(filename + filename_2 + "user_action_sequence_product_num.csv")
user_action_sequence_product_num_weight = pd.read_csv(filename + filename_2 + "user_action_sequence_product_num_weight.csv")

user_action_sequence_product_num_test = pd.read_csv(filename + filename_2_test + "user_action_sequence_product_num.csv")
user_action_sequence_product_num_weight_test = pd.read_csv(filename + filename_2_test + "user_action_sequence_product_num_weight.csv")

# Data of action sequence
user_action_sequence_data = pd.read_csv(filename + filename_2 + "user_action_sequence_oneweek.csv")

user_action_sequence_data_test = pd.read_csv(filename + filename_2_test + "user_action_sequence_oneweek.csv")

# user feature
user_feature_data=pd.read_csv(filename + filename_2 + 'user_feature_data.csv')

user_feature_data_test=pd.read_csv(filename + filename_2_test + 'user_feature_data.csv')

# product feature
item_feature_data=pd.read_csv(filename + filename_2 + 'item_feature_data.csv')

item_feature_data_test=pd.read_csv(filename + filename_2_test + 'item_feature_data.csv')

# user_item feature
user_item_feature_data=pd.read_csv(filename + filename_2 + 'user_item_feature_data.csv')

user_item_feature_data_test=pd.read_csv(filename + filename_2_test + 'user_item_feature_data.csv')

#%% input
# state_transform_matrix
input_state_matrix =  pd.merge(user_item_feature_data,item_feature_data,on='sku_id',how='left')

input_state_matrix_test =  pd.merge(user_item_feature_data_test,item_feature_data_test,on='sku_id',how='left')

# state_action_matrix
input_state_action_matrix =  pd.merge(user_item_feature_data,user_feature_data,on='user_id',how='left')

input_state_action_matrix_test =  pd.merge(user_item_feature_data_test,user_feature_data_test,on='user_id',how='left')

# for train
X_sa = input_state_action_matrix.iloc[:,2:].values   
X_s = input_state_matrix.iloc[:,2:].values 

# for test
X_sa_test = input_state_action_matrix_test.iloc[:,2:].values   
X_s_test = input_state_matrix_test.iloc[:,2:].values 

# for output evaluation
Y_output = user_action_sequence_product_num.iloc[:,2:3]
def action_sequence_transform(a):
    b=[]
    for ch in a:
        if ch=='p':
            b.append(0)
        elif ch=='f':
            b.append(1)
        elif ch=='c':
            b.append(2)
        elif ch=='o':
            b.append(3)
        elif ch=='-':
            b.append(4)
        else:
            b.append(5)
    b=np.array(b)
    return b

def sequence_transform(Y_output):
    max_seq = len(max(Y_output['action_sequence'].values.tolist(), key=len, default=''))
    Y_output_num = pd.DataFrame( [ [ None for i in range(max_seq) ] for j in range(Y_output.shape[0]) ] )
    for i in range(0,Y_output.shape[0]):
        a = copy.copy(action_sequence_transform(Y_output.iloc[i,0]))
        Y_output_num.iloc[i,0:a.shape[0]]=copy.copy(a)
    Y_output_num = Y_output_num.fillna(value=np.nan)
    return Y_output_num

Y_output = sequence_transform(Y_output)
Y_output.to_csv(filename + filename_2 + "action_sequence_transform.csv", index=False, sep=',', header=True)                
Y_output = pd.read_csv(filename + filename_2 + "action_sequence_transform.csv")

Y_output_test = user_action_sequence_product_num_test.iloc[:,2:3]
Y_output_test = sequence_transform(Y_output_test)
Y_output_test.to_csv(filename + filename_2_test + "action_sequence_transform.csv", index=False, sep=',', header=True)                
Y_output_test = pd.read_csv(filename + filename_2_test + "action_sequence_transform.csv")

#%% neural network

# transform to tensor
X_sa = torch.Tensor(X_sa)
X_s = torch.Tensor(X_s)
X_sa_test = torch.Tensor(X_sa_test)
X_s_test = torch.Tensor(X_s_test)

X = torch.cat((X_sa,X_s),dim=1)
X_test = torch.cat((X_sa_test,X_s_test),dim=1)

Y_output = torch.Tensor(Y_output.values)
Y_output_test = torch.Tensor(Y_output_test.values)


# dimension of the input parameter, the number of features
input_dim_sa = X_sa.shape[1]     
input_dim_s = X_s.shape[1] 
# dimensions of the output parameter
num_classes_sa = 7
num_classes_s = 6
# Hidden layer dimension, adjustable parameter
hidden_dim = 50 

# Regularization intensity, adjustable parameter
reg = 0.001              
#  learning rate of gradient descent, adjustable parameter
learning_rate = 0.001    
# momentum
momentum_value = 0.9 
# num of cycles
Epochs = 1000

Pi_0 = np.array([[1,0,0]])
Pi_0 = torch.Tensor(Pi_0).int()

class NN_HMM(nn.Module):
    def __init__(self):
        super(NN_HMM, self).__init__()
        self.state_action_M = nn.Sequential(
            nn.Linear(input_dim_sa, hidden_dim), 
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes_sa),
            nn.Softmax()
            )
        self.state_M = nn.Sequential(
            nn.Linear(input_dim_s, hidden_dim), 
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes_s),
            nn.Softmax()
            )
    def forward(self, X):
        X_sa = X.split(input_dim_sa,dim=1)[0]
        X_s = X.split(input_dim_sa,dim=1)[1]
        Y_sa = self.state_action_M(X_sa)
        Y_s = self.state_M(X_s)
        Y = torch.cat((Y_sa,Y_s),dim=1)
        return Y


model = NN_HMM()

class ForwardModel(nn.Module):
    def __init__(self,Pi_0,transition_probability,emission_probability):
        self.Pi_0 = Pi_0
        self.transition_probability = transition_probability
        self.emission_probability = emission_probability
        self.n_states = len(emission_probability)
    def log_prob(self,se):
        alpha = torch.zeros(self.n_states, ) 
        # initial value
        for i in range(self.n_states):
            alpha[i] = self.Pi_0[i]*self.emission_probability[i][se[0]]
        # recursion
        temp = torch.zeros(self.n_states, ) 
        temp_2 = torch.zeros(self.n_states, ) 
        temp_list = torch.zeros(self.n_states)
        for t in range(1,se.shape[0]):
            observation = se[t]            
            for i in range(self.n_states):
                
                for j in range(self.n_states):
                    x_1 = alpha[j].clone()
                    y_1 = self.transition_probability[j][i].clone()
                    temp_list[j] = x_1 * y_1
                temp[i] = torch.sum(temp_list)
                x = temp[i].clone()
                y = self.emission_probability[i][observation].clone()
                temp_2[i] = x * y
            alpha = temp_2.clone()
        # stop
        return -torch.log(torch.sum(alpha))
        #return torch.sum(alpha)

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Y, Y_output):
        #Y = outputs
        #Y_output = targets
        Y_state_action = Y.split(num_classes_sa,dim=1)[0]
        Y_state = Y.split(num_classes_sa,dim=1)[1]   
        loss_list = torch.zeros(Y.shape[0])
        for i in range(0,Y.shape[0]):
            print('i: ', i)
            Matrix_state_action = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
            Matrix_state_action = torch.Tensor(Matrix_state_action)
            Matrix_state_action[0,3] = 1-Y_state_action[i,0]-Y_state_action[i,1]-Y_state_action[i,2]
            Matrix_state_action[0,0:3] = Y_state_action[i,0:3]
            Matrix_state_action[1,4] = Y_state_action[i,3]
            Matrix_state_action[1,5] = 1- Y_state_action[1,3]
            Matrix_state_action[2,0:3] = Y_state_action[i,4:7]
            Matrix_state_action[2,3] = 1 - Y_state_action[i,4] - Y_state_action[i,5] - Y_state_action[i,6]
                        
            Matrix_state = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
            Matrix_state = torch.Tensor(Matrix_state)
            Matrix_state[:,0:2] = Y_state[i,:].reshape(3,2)
            Matrix_state[0,2] = 1 - Y_state[i,0] - Y_state[i,1] 
            Matrix_state[1,2] = 1 - Y_state[i,2] - Y_state[i,3] 
            Matrix_state[2,2] = 1 - Y_state[i,4] - Y_state[i,5] 
            
            emission_probability = Matrix_state_action.clone()
            transition_probability = Matrix_state.clone()
            model_hmm = ForwardModel(Pi_0,transition_probability,emission_probability)
        
            se_temp = Y_output[i]
            se = se_temp[0:torch.where(torch.isnan(se_temp))[0].min().item()].int()
            # se = torch.tensor([0,5])
            a = model_hmm.log_prob(se)
            # a.backward(retain_graph=True)
            loss_list[i] = a 
        sum_loss = torch.sum(loss_list)
        loss_hmm = sum_loss/Y_output.shape[0]
        # loss_hmm.backward(retain_graph=True)
        return loss_hmm        
    
criterion = My_loss()

#  iterative optimization algorithm, stochastic gradient descent algorithm
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=reg, momentum=momentum_value)  

loss_dict = []
# Iterative training
torch.autograd.set_detect_anomaly(True)
start_read_train = time.perf_counter()
#def closure():
#    optimizer.zero_grad()
#    outputs = model(inputs)
#    loss = criterion(outputs, targets)
#    loss.backward(retain_graph=True)
#    return loss
for epoch in range(Epochs):
    inputs = X
    targets = Y_output

    # Forward propagation computes the output of the network structure
    outputs = model(inputs)
    # Compute loss
    loss = criterion(outputs, targets)
    # Back propagation updates parameters
    optimizer.zero_grad()  # gradient is zeroed out for each iteration
    loss.backward(retain_graph=True)
    optimizer.step()  # update weights

    #optimizer.step(closure)
    
    loss_dict.append(loss.item())
    #loss_dict.append(closure().item())
    #if epoch % 5 == 0:   
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, Epochs, loss.item()))
    #print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, Epochs, closure().item()))
elapsed_read_train = (time.perf_counter() - start_read_train) 
print("The data for training： ",elapsed_read_train)  
# plot loss
plt.plot(loss_dict, label='loss for every epoch')
plt.text(1, loss_dict[0]-0.0001, '%.4f'%loss_dict[0], ha='center', va='bottom', fontsize=10.5)
plt.text(Epochs, loss_dict[Epochs-1]+0.0001, '%.4f'%loss_dict[Epochs-1], ha='center', va='bottom', fontsize=10.5)
plt.legend()
plt.show()

#save
PATH_1 = filename + "SaveModel/Model.tar"
torch.save(model, PATH_1)
PATH_2 = filename + "SaveModel/model_param.pkl"
torch.save(model.state_dict(), PATH_2)
#load
model = torch.load(PATH_1)
model.state_dict()



inputs_test = X_test
targets_test = Y_output_test
outputs_test = model(inputs_test)
loss_test = criterion(outputs_test, targets_test) 
print('Loss_test is {}:'.format(loss_test))  

def calculate_buy(matrix_s,matrix_sa):
    # The state probability at time t can be obtained from the initial state and state transition matrix
    # The behavior probability at time T can be calculated
    # If the probability of buying is greater than 0.7, it means buying 
    # If the probability of buying is lower than 0.3, it means not buying
    T = 5
    if_buy = 0
    for t in range(T):
        matrix_s_t = torch.dot(Pi_0,matrix_s)
        for tn in range(1,t+1):
            matrix_s_t = torch.matmul(matrix_s_t,matrix_s)
        action_prob = torch.matmul(matrix_s_t,matrix_sa)
        order_prob = action_prob[3].item()
        if order_prob >= 0.7:
            if_buy = 1
            return if_buy
        elif order_prob <=0.3:
            if_buy = 0
    return if_buy

# The prediction accuracy is obtained by comparing the real purchase with the predicted purchase
Y_sa_test = outputs_test.split(num_classes_sa,dim=1)[0]
Y_s_test = outputs_test.split(num_classes_sa,dim=1)[1]   
sum = 0
for i in range(0,outputs_test.shape[0]):
    Matrix_sa_test=np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    Matrix_sa_test=torch.Tensor(Matrix_sa_test)
    Matrix_sa_test[0,3]=1-Y_sa_test[i,0]-Y_sa_test[i,1]-Y_sa_test[i,2]
    Matrix_sa_test[0,0:3]=Y_sa_test[i,0:3]
    Matrix_sa_test[1,4]=Y_sa_test[i,3]
    Matrix_sa_test[1,5]=1-Y_sa_test[1,3]
    Matrix_sa_test[2,0:3]=Y_sa_test[i,4:7]
    Matrix_sa_test[2,3]=1-Y_sa_test[i,4]-Y_sa_test[i,5]-Y_sa_test[i,6]
        
    Matrix_s_test=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    Matrix_s_test=torch.Tensor(Matrix_s_test)
    Matrix_s_test[:,0:2]=Y_s_test[i,:].reshape(3,2)
    Matrix_s_test[:,2]=1-Matrix_s_test[:,0]-Matrix_s_test[:,1]  
    if_buy_test = calculate_buy(Matrix_s_test,Matrix_sa_test)
    if 3 in Y_output_test[[i]]:
        if_buy_real = 1
    else:
        if_buy_real = 0
    if if_buy_test == if_buy_real:
        sum += 1
if_buy_predict_prob = sum/outputs_test.shape[0]
print('The probability of true prediction for buying or not buying is: {}'.format(if_buy_predict_prob))  

