#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import os
from Scripts.createDataset import getData, getDataLoaders, loadData
from Scripts.network import network
from Scripts.training import trainModel
from Scripts.plotting import compute_errors, plotTestResults


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


torch.manual_seed(1)
np.random.seed(1)


# In[ ]:


# decide case
_,trajectories = loadData()
shuffle_idx = np.random.permutation(len(trajectories))
trajectories = trajectories[shuffle_idx]
number_samples,number_components = trajectories.shape


# In[ ]:


L = 3.3


# In[ ]:


#Importing and shuffling the trajectories

number_samples,number_components = trajectories.shape
#Randomize the order of the trajectories
indices = np.random.permutation(len(trajectories))
trajectories = trajectories[indices]

number_elements = int(number_components/4)-1
data_train, data_test, data_val, x_train, x_test, x_val, y_train, y_test, y_val = getData(number_elements,number_samples,trajectories)


# In[ ]:


training_trajectories = np.concatenate((x_train[:,:4],y_train,x_train[:,-4:]),axis=1)
test_trajectories = np.concatenate((x_test[:,:4],y_test,x_test[:,-4:]),axis=1)
val_trajectories = np.concatenate((x_val[:,:4],y_val,x_val[:,-4:]),axis=1)


# In[ ]:


torch.manual_seed(1)
np.random.seed(1)


# In[ ]:


impose_bcs = input("Want to impose the boundary conditions? Choose among 'Yes' and 'No'")=="Yes"
pre_trained = input("Want to work with a pre-trained model? Choose among 'Yes' and 'No'")=="Yes"


# In[ ]:


params = {}

if pre_trained:
    if impose_bcs:
        params = {'act': 'tanh',
                 'n_layers': 8,
                 'hidden_nodes': 58,
                 'networkarch': 0,
                 'lr': 5e-3,
                 'weight_decay': 0}
    else:
        params = {'act': 'tanh',
                 'n_layers': 8,
                 'hidden_nodes': 93,
                 'networkarch': 0,
                 'lr': 5e-3,
                 'weight_decay': 0}
        
if params=={}:
    print("No parameters have been specified. Let's input them:\n\n")
    act = input("What activation function to use? Choose among 'sin', 'sigmoid', 'swish', 'tanh' ")
    nlayers = int(input("How many layers do you want the network to have? "))
    hidden_nodes = int(input("How many hidden nodes do you want the network to have? "))

    lr = float(input("What learning rate do you want to use? "))
    weight_decay = float(input("What weight decay do you want to use? "))
    networkarch = int(input("Network architecture: Type 0 for MULT, 1 for ResNet, 2 for MLP: "))
    params = {"act": act,
              "n_layers":nlayers,
              "hidden_nodes":hidden_nodes,
              "lr":lr,
              "weight_decay":weight_decay,
              "networkarch":networkarch}

act = params["act"]
nlayers = params["n_layers"]
hidden_nodes = params["hidden_nodes"]

netarch = params["networkarch"]
if netarch == 0:
    is_deeponet = True
    is_res = False
elif netarch == 1:
    is_deeponet = False
    is_res = True
else:
    is_deeponet = False
    is_res = False

model = network(impose_bcs=impose_bcs,act_name=act, nlayers=nlayers, hidden_nodes = hidden_nodes, is_deeponet=is_deeponet, is_res=is_res)
model.to(device);
batch_size = 1024
trainloader, testloader, valloader = getDataLoaders(batch_size,data_train,data_test,data_val,type='regression')

if pre_trained:
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/ContinuousNetworkTheta/TrainedModels")
    
    if impose_bcs:
        model.load_state_dict(torch.load("BcsTrainedModel.pt",map_location=device))
    else:
        model.load_state_dict(torch.load("noBcsTrainedModel.pt",map_location=device))
    os.chdir(original_dir)
else:
    weight_decay = params["weight_decay"]
    lr = params["lr"]
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    criterion = nn.MSELoss()
    epochs = 100
    print("Now the training has started")
    loss = trainModel(number_elements,device,model,criterion,optimizer,epochs,trainloader,valloader)


# In[ ]:


model.eval();
res_derivative,theta = plotTestResults(model,device,number_elements,number_components,x_train,x_test,y_train,y_test)


# In[ ]:


test_error, train_error, val_error, pred_train_all, pred_test_all, pred_val_all = compute_errors(model,device,number_elements,number_components,x_train,x_test,x_val,y_train,y_test,y_val)


# In[ ]:


print(f"Train error is {train_error}, validation error is {val_error}, and test error is {test_error}.")

