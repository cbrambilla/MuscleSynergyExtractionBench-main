import numpy as np
import utils as ut
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
#------------------------------------------------------------------

n_components = 6    #number of synergies
planes = ["f","h","l","r","u"]  #planes of movements
S = "00"    #subject ID
nb_iter = 20

#----------------------------------------------------

for i in range(len(planes)):
    mat_contents = loadmat(os.path.join(os.getcwd(),'S00_input.mat'))
    sorted(mat_contents.keys())
    xtrain = mat_contents['train_export_%s'%planes[i]]
    xtest = mat_contents['test_export_%s'%planes[i]]

    #---------------------------------------------------------------

    autoencoder,syns,tcoef = ut.setup_AE(n_components, xtrain, xtest)
    reconstruct_AE = autoencoder.predict([xtest[0].reshape(-1,900,16)])[0]
    #----------------------------------------------------

    model = NMF(n_components=n_components, init='random',solver='cd',max_iter = 500)
    model.fit(abs(xtrain.reshape(-1, xtrain.shape[-1])))
    H = model.components_

    
    W_test = 0
    mse_NNMF = 0
    r_squared_NNMF = 0

    for j in range(nb_iter):
        W_test_temp = model.transform(abs(xtest[0]))
        H_norm_temp,W_test_norm_temp = ut.normalize_syns(H,W_test_temp)
        reconstruct_NNMF_temp = np.matmul(W_test_norm_temp,H_norm_temp)
        mse_NNMF_temp = mean_squared_error(reconstruct_NNMF_temp, xtest[0])
        r_squared_NNMF_temp = ut.r_squared(reconstruct_NNMF_temp, xtest[0])

        if r_squared_NNMF_temp > r_squared_NNMF:
            mse_NNMF = mse_NNMF_temp
            r_squared_NNMF = r_squared_NNMF_temp
            W_test = W_test_norm_temp
            H = H_norm_temp


    #-------------------------------------------------
    # Compute the scores
    #-------------------------------------------------

    mse_AE = mean_squared_error(reconstruct_AE,xtest[0])
    print(str(mse_NNMF)+",")
    print(str(mse_AE)+",")

    r_squared_AE = ut.r_squared(reconstruct_AE, xtest[0])
    print(str(r_squared_NNMF)+",")
    print(str(r_squared_AE )+",") 


    #---------------------------------------------------------------------------
    #   synergy similarity and tcoef correlations
    #----------------------------------------------------------------------------

    new_synAE,new_tAE,new_synNNMF,new_tNNMF, similarities = ut.match_synergies(syns,tcoef,H,W_test,n_components)
    print(str(similarities)+",")
    corrs = ut.tcoef_correlations(new_tAE, new_tNNMF, n_components)
    print(str(corrs)+",")


    obj = ut.MyClass(syns)
    ut.save_object(obj,"syns_AE_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(tcoef)
    ut.save_object(obj,"tcoef_AE_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(mse_NNMF)
    ut.save_object(obj,"mse_NNMF_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(r_squared_NNMF)
    ut.save_object(obj,"r_squared_NNMF_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(mse_AE)
    ut.save_object(obj,"mse_AE_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(r_squared_AE)
    ut.save_object(obj,"r_squared_AE_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(similarities)
    ut.save_object(obj,"sims_single_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(corrs)
    ut.save_object(obj,"corrs_single_S%s_%s" % (S,planes[i]))

