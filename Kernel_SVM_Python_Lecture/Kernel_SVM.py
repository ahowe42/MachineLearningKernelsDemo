import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression

'''
http://scikit-learn.org/stable/modules/metrics.html#
http://scikit-learn.org/stable/modules/svm.html#
http://scikit-learn.org/stable/modules/cross_validation.html#
http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#
'''

# get the data
iris = load_iris()
data = iris.data
labels = iris.target
(n,p) = data.shape

''' Evaluate performance in original data space with Logistic Regression '''
LR = LogisticRegression()
class_rate = LR.fit(X = data, y = labels).score(X = data, y = labels)
print('Logistic Regression: %0.2f%%'%(100*class_rate))


''' Now evaluate performance in Feature Space with 4 kernels '''
kerns = ['linear','quadratic','rbf','sigmoid']
kerncnt = len(kerns)

# homogenous linear SVC
line = svm.SVC(kernel='linear')
class_rate = line.fit(X = data, y = labels).score(X = data, y = labels)
print('Linear Kernel: %0.2f%%'%(100*class_rate))
# homogenous quadratic SVC
quad = svm.SVC(kernel='poly',degree=2,gamma=1/p,coef0=0)
class_rate = quad.fit(X = data, y = labels).score(X = data, y = labels)
print('Quadratic Kernel: %0.2f%%'%(100*class_rate))
# RBF SVC
rbfn = svm.SVC(kernel='rbf',gamma=1/p)
class_rate = rbfn.fit(X = data, y = labels).score(X = data, y = labels)
print('RBF Kernel: %0.2f%%'%(100*class_rate))
# sigmoid SVC
sigm = svm.SVC(kernel='sigmoid',gamma=1/p,coef0=0)
class_rate = sigm.fit(X = data, y = labels).score(X = data, y = labels)
print('Sigmoid Kernel: %0.2f%%'%(100*class_rate))


''' Overfitting to the observed data is an important concern, so use cross validation '''
# set up the randomized cross-validator
CVs = 100
cvrand = cross_validation.ShuffleSplit(n, n_iter = CVs, train_size = 0.6,\
    test_size = 0.4, random_state = 12272010)
cv_kern_scores = np.zeros((CVs,kerncnt),dtype=float)

# homogenous linear SVC
cv_kern_scores[:,0] = cross_validation.cross_val_score(line, data, labels,\
    cv=cvrand)
# homogenous quadratic SVC
cv_kern_scores[:,1] = cross_validation.cross_val_score(quad, data, labels,\
    cv=cvrand)
# RBF SVC
cv_kern_scores[:,2] = cross_validation.cross_val_score(rbfn, data, labels,\
    cv=cvrand)
# sigmoid SVC
cv_kern_scores[:,3] = cross_validation.cross_val_score(sigm, data, labels,\
    cv=cvrand)

# summarize
scores = pd.DataFrame(data = cv_kern_scores,columns = kerns)
scores.describe()


''' Instead of arbitrarily setting the parameters, let's do a grid search among reasonable alternatives '''
# seach the subspace of possible parameters
params = [{'kernel':['poly'],'degree':[1,2,3],'gamma':[1/p,1,2],'coef0':[-1,0,1]},\
	{'kernel':['rbf'],'gamma':[1/p,1,2],'degree':[3],'coef0':[0]},\
	{'kernel':['sigmoid'],'gamma':[1/p,1,2],'coef0':[-1,0,1],'degree':[3]}]
GSC = grid_search.GridSearchCV(estimator = svm.SVC(), param_grid = params,\
    cv = cvrand, n_jobs = -1)
GSC.fit(X = data, y = labels)

# print the results
print('Best Model Score = %0.2f%%'%(100*GSC.best_score_))
for param in GSC.best_params_.keys():
	print('\t%s = %r'%(param,GSC.best_params_[param]))
	
# now run the best model using cross-validation & summarize it's performance
best_mod = svm.SVC(kernel=GSC.best_params_['kernel'],\
    gamma=GSC.best_params_['gamma'],coef0=GSC.best_params_['coef0'],\
    degree=GSC.best_params_['degree'])
best_scores = cross_validation.cross_val_score(best_mod, data, labels,\
    cv=cvrand)
pd.Series(data = best_scores, name = GSC.best_params_['kernel']).describe()
	

''' To be able to select the best subset of features, run everything seen on all subsets '''
# generate all subsets ...
bins = VarSubset(p)[0][1:]
subset_cnt = len(bins)
best_mods = []
best_scs = [0]*subset_cnt
# ... and run the GSC
for ind,sub in enumerate(bins):
    print('Subset %r (%d)'%(sub,ind))
    # cross-validated grid search
    GSC.fit(X = data[:,sub], y = labels)
    # save the best model & score per subset
    best_mods.append(GSC.best_params_)
    best_scs[ind] = GSC.best_score_
# now find the subset with the best score and run the model
bst = np.argmax(best_scs)
best_model = best_mods[bst]
best_subset = bins[bst,:]
print('Best Subset of Features: %r\nBest Kernel SVM Model: %r'%\
    (best_subset,best_model))
# now run the best model using cross-validation & summarize it's performance
best_mod = svm.SVC(kernel=best_model['kernel'],\
    gamma=best_model['gamma'],coef0=best_model['coef0'],\
    degree=best_model['degree'])
best_scores = cross_validation.cross_val_score(best_mod, data[:,best_subset],\
    labels,cv=cvrand)
pd.Series(data = best_scores, name = GSC.best_params_['kernel']).describe()


def VarSubset(p):
	"""
	Generate an array of binary indices that can be used for all-subset combinatorial analysis
	of a dataset with p variables.
	---
	Usage: subset_binaries, subset_sizes = VarSubset(p)
	---
	p: integer indicating number of variables to subset
	subset_binaries: (2^p, p) array of all subsets binary indices that can be used to subset
		into the presumed original data matrix
	subset_sizes: 2^p array indicating number of variables in each subset
	---
	ex: p = 4; cols = np.arange(p); bins,sizs = QB.VarSubset(p); print(cols[bins[8,:]])
	JAH 20121018
	"""
	
	# check that p is int; could just duck-type it, but if user passes something else, something is screwed up
	if type(p) is not int:
		raise ValueError("The number variables must be integer: %s"%VarSubset.__doc__)
	
	# prepare the output array; we want bool, but have to start with int, so the assignment below works correctly
	subbins = np.zeros((2**p,p),dtype=int)
	
	# loop through all subsets :-( getting the binary representations
	for cnt in range(1,2**p):
		# get binary representation into a list, then put it in the array
		tmp = bin(cnt)[2:]
		subbins[cnt,(-len(tmp)):] = list(tmp)
	
	# fill in the variable counts
	subsize = np.sum(subbins,axis=1)
	
	# finally sort by variable counts
	tmp = np.argsort(subsize)
	
	return subbins[tmp,:]==1, subsize[tmp]