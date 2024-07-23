import numpy as np
import utils as ut
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
#------------------------------------------------------------------

mat_contents = loadmat(os.path.join(os.getcwd(),'S00_input.mat'))
sorted(mat_contents.keys())
xtrain_f = mat_contents['train_export_f']
xtest_f = mat_contents['test_export_f']
xtrain_h = mat_contents['train_export_h']
xtest_h = mat_contents['test_export_h']
xtrain_u = mat_contents['train_export_u']
xtest_u = mat_contents['test_export_u']
xtrain_r = mat_contents['train_export_r']
xtest_r = mat_contents['test_export_r']
xtrain_l = mat_contents['train_export_l']
xtest_l = mat_contents['test_export_l']

#---------------------------------------------------------------
xtrain = np.concatenate((xtrain_f,xtrain_h,xtrain_u,xtrain_r,xtrain_l))
xtest = np.concatenate((xtest_f, xtest_h, xtest_u, xtest_r, xtest_l))

#----------------------------------------------------
n_components = 6    #number of synergies
planes = ["f","h","u","r","l"]  #planes of movement
S = "00"    #subject ID
#---------------------------------------------------------------

# autoencoder,syns,tcoef = ut.setup_AE(n_components, xtrain, xtest)
autoencoder,syns,tcoef = ut.setup_AE_all_planes(n_components, xtrain, xtest)

model = NMF(n_components=n_components, init='random',solver='cd',max_iter = 500)
model.fit(abs(xtrain.reshape(-1, xtrain.shape[-1])))
H = model.components_

reconstruct_AE = []
reconstruct_NNMF = []
W_test = []

for i in range(5):
    reconstruct_AE_f = autoencoder.predict([xtest[i].reshape(-1,900,16)])[0]
    reconstruct_AE.append(reconstruct_AE_f)

    W_test_f = model.transform(abs(xtest[i]))
    H_norm,W_test_norm = ut.normalize_syns(H,W_test_f)
    W_test.append(W_test_norm)
    reconstruct_NNMF_f = np.matmul(W_test_norm,H_norm)
    reconstruct_NNMF.append(reconstruct_NNMF_f)


#-------------------------------------------------
# Compute the scores
#-------------------------------------------------

mse_AE = []
mse_NNMF = []
r_squared_AE = []
r_squared_NNMF =[]

for i in range(5):
    
    mse_NNMF_f = mean_squared_error(reconstruct_NNMF[i], xtest[i])
    mse_NNMF.append(mse_NNMF_f)
    mse_AE_f = mean_squared_error(reconstruct_AE[i],xtest[i])
    mse_AE.append(mse_AE_f)
    print('MSE_NMF_%s: ' %planes[i] + str(mse_NNMF_f)+",")
    print('MSE_AE_%s: ' %planes[i] + str(mse_AE_f)+",")

    r_squared_NNMF_f = ut.r_squared(reconstruct_NNMF[i], xtest[i])
    r_squared_NNMF.append(r_squared_NNMF_f)
    r_squared_AE_f = ut.r_squared(reconstruct_AE[i], xtest[i])
    r_squared_AE.append(r_squared_AE_f)
    print('R2_NMF_%s: ' %planes[i] + str(r_squared_NNMF_f)+",")
    print('R2_AE_%s: ' %planes[i] + str(r_squared_AE_f)+",") 



#---------------------------------------------------------------------------
#   synergy similarity and tcoef correlations
#----------------------------------------------------------------------------

new_syns_AE = []
new_tcoef_AE = []
new_syns_NNMF = []
new_tcoef_NNMF = []
sims = []
corrs = []

for i in range(5):

    new_synAE_f,new_tAE_f,new_synNNMF_f,new_tNNMF_f, similarities_f = ut.match_synergies(syns[i],tcoef[i],H,W_test[i],n_components)
    corrs_f = ut.tcoef_correlations(new_tAE_f, new_tNNMF_f, n_components)
    print('similarities_%s: ' %planes[i] + str(similarities_f)+",")
    print('correlations_%s: ' %planes[i] + str(corrs_f)+",") 
    new_syns_AE.append(new_synAE_f)
    new_tcoef_AE.append(new_tAE_f)
    new_syns_NNMF.append(new_synNNMF_f)
    new_tcoef_NNMF.append(new_tNNMF_f)
    sims.append(similarities_f)
    corrs.append(corrs_f)


#---------------------------------------------------------------------------
#   Save data
#----------------------------------------------------------------------------

for i in range(5):

    obj = ut.MyClass(new_syns_AE[i])
    ut.save_object(obj,"syns_AE_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(new_tcoef_AE[i])
    ut.save_object(obj,"tcoef_AE_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(new_syns_NNMF[i])
    ut.save_object(obj,"syns_NNMF_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(new_tcoef_NNMF[i])
    ut.save_object(obj,"tcoef_NNMF_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(mse_AE[i])
    ut.save_object(obj,"mse_AE_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(r_squared_AE[i])
    ut.save_object(obj,"r_squared_AE_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(mse_NNMF[i])
    ut.save_object(obj,"mse_NNMF_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(r_squared_NNMF[i])
    ut.save_object(obj,"r_squared_NNMF_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(sims[i])
    ut.save_object(obj,"sims_multi_S%s_%s" % (S,planes[i]))

    obj = ut.MyClass(corrs[i])
    ut.save_object(obj,"corrs_multi_S%s_%s" % (S,planes[i]))



#---------------------------------------------------------------------------
#   Reconstruction plots
#----------------------------------------------------------------------------

# plt.figure(1)
# ut.plot_syns(new_syns_AE[0], new_tcoef_AE[0], 'AE',n_components)
# plt.figure(2)
# ut.plot_syns(H, W_test, 'NNMF',n_components)
# plt.figure(3)
# ut.plot_reconstr(reconstruct_AE, xtest, 'AE')
# plt.figure(4)
# ut.plot_reconstr(reconstruct_NNMF, xtest, 'NNMF')

