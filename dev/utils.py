import numpy as np

def print_layer_information(rl_net):
    for i, submodule in enumerate(rl_net.net.modules()):
        print(i, submodule)

def convert_single_layer(ind,activations):
    shp=activations[ind][0].shape
    t=len(activations[ind])
    ts=np.zeros((np.prod(shp),t))
    for j in range(0,t):
        ts[:,j]=activations[ind][j].flatten()
    return ts

def convert_multiple_layers(inds,activations):
    t=len(activations[inds[0]])
    ts=np.zeros((1,t))
    for j in inds:
        ts_=convert_single_layer(j,activations)
        ts=np.vstack((ts,ts_))
    ts=ts[1:,:]
    return ts

def one_hot_action(actions_record,action_label):
    action_one_hot=[]
    for j in actions_record:
        if j==action_label:
            action_one_hot.append(1)
        else:
            action_one_hot.append(0)
    return action_one_hot

def one_hot_action_arr(actions_record):
    action_one_hot_arr=[]
    for j in range(0,5):
        action_one_hot_arr.append(one_hot_action(actions_record,j))
    return np.array(action_one_hot_arr)
