from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def compute_rfield(imgs,V):
    imgs=imgs.T
    lam=10
    npix=84*84
    rf_weights = np.linalg.solve((imgs @ imgs.T + lam * np.eye(npix)),  (imgs @ V)).reshape(84,84,V.shape[1])
    return rf_weights

def fit_ep(ts,n_ensembles=100,lam=0.1):
    activations=ts.T
    ep=EnsemblePursuit(n_components=n_ensembles,n_kmeans=n_ensembles,lam=0.1)
    ep.fit(activations)
    return ep.components_, ep.weights

def fit_pca(ts,n_components=100):
    pca=PCA(n_components=n_components)
    V=pca.fit_transform(ts.T)
    return V

def plot_receptive_fields(V,U,imgs):
    rf_weights=compute_rfield(imgs,V)
    for j in range(0,V.shape[1]):
        n_neurons=np.nonzero(U[:,j].flatten())[0].shape[0]
        plt.imshow(rf_weights[:,:,j],cmap='bwr')
        plt.title('Ensemble '+str(j)+', n_neurons='+str(n_neurons))
        plt.show()

def plot_behavior_receptive_fields(action_one_hot_arr,imgs):
    rf_weights=compute_rfield(imgs,action_one_hot_arr)
    for j in range(0,action_one_hot_arr.shape[1]):
        plt.imshow(rf_weights[:,:,j],cmap='bwr')
        plt.title('Action rf, action: '+str(j))
        plt.show()
