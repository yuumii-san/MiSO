import numpy as np
import scipy.linalg as slin
import sklearn.model_selection as ms

# imports for parallization capabilities
from joblib import Parallel,delayed
from functools import partial
from psutil import cpu_count

class factor_analysis:


    def __init__(self,model_type='fa',min_var=0.01):
        if model_type.lower()!='ppca' and model_type.lower()!='fa':
            raise ValueError('model_type should be "fa" or "ppca"')
        self.model_type = model_type
        self.min_var = min_var


    def train(self,X,zDim,tol=1e-6,max_iter=int(1e8),verbose=False,rand_seed=None,X_early_stop=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        early_stop = not(X_early_stop is None)

        # some useful parameters
        N,D = X.shape
        mu = X.mean(axis=0)
        cX = X - mu
        covX = (1/N)*(cX.T.dot(cX))#(np.cov(cX.T,bias=True)
        if early_stop:
            cX_test = X_early_stop-mu
            cov_Xtest = (1/N)*(cX_test.T.dot(cX_test))
        var_floor = self.min_var*np.diag(covX)
        Iz = np.identity(zDim)
        Ix = np.identity(D)
        const = D*np.log(2*np.pi)

        # check that the data matrix is full rank
        if np.linalg.matrix_rank(covX)==D:
            scale = np.exp(2/D*np.sum(np.log(np.diag(np.linalg.cholesky(covX)))))
        else:
            raise np.linalg.LinAlgError('Covariance matrix is low rank')
        
        # initialize fa parameters (L and Ph)
        Ph = np.diag(covX)
        # return mu and Ph if zDim=0
        if zDim==0:
            L = np.random.randn(D,zDim)
            fa_params = {'mu':mu,'L':L,'Ph':Ph,'zDim':zDim}
            self.fa_params = fa_params
            z,LL = self.estep(X)
            return LL
        else:
            L = np.random.randn(D,zDim) * np.sqrt(scale/zDim)

        # fit fa model
        LL = []
        testLL = []
        for i in range(max_iter):
            # E-step
            iPh = np.diag(1/Ph)
            iPhL = iPh.dot(L)
            iSig = iPh - iPhL.dot(slin.inv(Iz+(L.T).dot(iPhL))).dot(iPhL.T)
            iSigL = iSig.dot(L)
            cov_iSigL = covX.dot(iSigL)
            E_zz = Iz - (L.T).dot(iSigL) + (iSigL.T).dot(cov_iSigL)

            # M-step
            L = cov_iSigL.dot(slin.inv(E_zz))
            Ph = np.diag(covX) - np.diag(cov_iSigL.dot(L.T))

            # set noise values to be the same if ppca
            if self.model_type.lower()=='ppca':
                Ph = np.mean(Ph) * np.ones(Ph.shape)

            # make sure values are above noise floor
            Ph = np.maximum(Ph,self.min_var)

            # compute log likelihood
            logDet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(iSig))))
            curr_LL = -N/2 * (const - logDet + np.trace(iSig.dot(covX)))
            LL.append(curr_LL)
            if early_stop:
                curr_testLL = -N/2 * (const - logDet + np.trace(iSig.dot(cov_Xtest)))
                testLL.append(curr_testLL)
            if verbose:
                print('EM iteration ',i,', LL={:.2f}'.format(curr_LL))

            # check convergence (training LL increases by less than tol, or testLL decreases)
            if i>1:
                if (LL[-1]-LL[-2])<tol or (early_stop and curr_testLL<testLL[-2]):
                    break

        # store model parameters in dict
        fa_params = {'mu':mu,'L':L,'Ph':Ph,'zDim':zDim}
        self.fa_params = fa_params

        return np.array(LL), np.array(testLL)


    def train_earlyStop(self,X,zDim,n_folds=10,rand_seed=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
            
        N,D = X.shape

        # create k-fold iterator
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        self.earlyStop_params = []
        for train_idx,test_idx in cv_kfold.split(X):
            X_train,X_test = X[train_idx], X[test_idx]
            tmp_mdl = factor_analysis(model_type=self.model_type,min_var=self.min_var)
            tmp_mdl.train(X_train,zDim,rand_seed=rand_seed,X_early_stop=X_test)
            self.earlyStop_params.append(tmp_mdl.get_params())

        return 


    def get_params(self):
        return self.fa_params


    def set_params(self,fa_params):
        self.fa_params = fa_params


    def estep(self,X):
        mu = self.fa_params['mu']
        L = self.fa_params['L']
        Ph = self.fa_params['Ph']
        zDim = self.fa_params['zDim']

        N,xDim = X.shape
        cX = X - mu
        covX = (1/N) * ((cX.T).dot(cX))
        Iz = np.eye(zDim)

        est_cov = L.dot(L.T) + np.diag(Ph)
        iSig = slin.inv(est_cov)

        # compute posterior on latents
        z_mu = (L.T).dot(iSig).dot(cX.T)
        z_cov = Iz - (L.T).dot(iSig).dot(L)

        # compute LL
        const = xDim*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(np.linalg.cholesky(iSig))))
        LL = -N/2 * (const - logDet + np.trace(iSig.dot(covX)))

        # return posterior and LL
        z = { 'z_mu':z_mu.T, 'z_cov':z_cov }
        return z,LL


    def orthogonalize(self,z_mu):
        N,zDim = z_mu.shape

        L = self.fa_params['L']
        u,s,vt = np.linalg.svd(L,full_matrices=False)
        Lorth = u
        TT = np.diag(s).dot(vt)
        z_orth = (TT.dot(z_mu.T)).T

        return z_orth, Lorth


    def crossvalidate(self,X,zDim_list=np.linspace(0,10,11),n_folds=10,tol=1e-6,verbose=True,rand_seed=None,parallelize=False,early_stop=False):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
            
        N,D = X.shape

        # make sure z dims are integers
        z_list = zDim_list.astype(int)

        # create k-fold iterator
        if verbose:
            print('Crossvalidating FA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0            
        LLs = np.zeros([n_folds,len(z_list)])
        for train_idx,test_idx in cv_kfold.split(X):
            if verbose:
                print('   Fold ',i+1,' of ',n_folds,'...')

            X_train,X_test = X[train_idx], X[test_idx]
            
            # iterate through each zDim
            if parallelize:
                func = partial(self._cv_helper,Xtrain=X_train,Xtest=X_test,tol=tol,rand_seed=rand_seed,early_stop=early_stop)
                tmp_LL = Parallel(n_jobs=cpu_count(logical=False),backend='loky')\
                    (delayed(func)(z_list[j]) for j in range(len(z_list)))
                LLs[i,:] = tmp_LL
            else:
                for j in range(len(z_list)):
                    LLs[i,j] = self._cv_helper(z_list[j],X_train,X_test,tol=tol,rand_seed=rand_seed,early_stop=early_stop)
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)

        # find the best # of z dimensions and train FA model
        max_idx = np.argmax(sum_LLs)
        self.train(X,z_list[max_idx])

        LL_curves = {
            'z_list':z_list,
            'LLs':sum_LLs,
            'final_LL':sum_LLs[max_idx],
            'zDim':z_list[max_idx]
        }

        return LL_curves


    def _cv_helper(self,zDim,Xtrain,Xtest,tol=1e-6,rand_seed=None,early_stop=False):
        tmp = factor_analysis(self.model_type,self.min_var)
        if early_stop:
            tmp.train(Xtrain,zDim,tol=tol,rand_seed=rand_seed,X_early_stop=Xtest)
        else:
            tmp.train(Xtrain,zDim,tol=tol,rand_seed=rand_seed)
        z,curr_LL = tmp.estep(Xtest)
        return curr_LL


    def compute_psv(self):
        L,Ph = self.fa_params['L'], self.fa_params['Ph']
        shared_var = np.diag(L.dot(L.T))
        psv = shared_var / (shared_var+Ph)
        avg_psv = np.mean(psv)
        return avg_psv, psv


    def compute_psv_heldout(self,X_heldout):
        L,Ph = self.fa_params['L'], self.fa_params['Ph']
        zDim = L.shape[1]

        sig = L.dot(L.T)+np.diag(Ph)
        iSig = slin.inv(sig)
        covX = np.cov(X_heldout.T,bias=True)

        # z_unc = np.eye(zDim) - L.T.dot(iSig).dot(L)
        z_unc = np.eye(zDim) - L.T.dot(iSig).dot(L)
        cov_ez = L.T.dot(iSig).dot(covX).dot(iSig).dot(L)

        shared_var = np.diag(L.dot(z_unc + cov_ez).dot(L.T))
        total_var = np.diag(covX)
        return np.mean(shared_var/total_var)


    def compute_dshared(self,cutoff_thresh=0.95):
        L = self.fa_params['L']
        shared_cov = L.dot(L.T)
        u,s,vt = np.linalg.svd(shared_cov,full_matrices=False,hermitian=True)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        return dims[0]+1
        

    def compute_part_ratio(self):
        L = self.fa_params['L']
        shared_cov = L.dot(L.T)
        u,s,vt = np.linalg.svd(shared_cov,full_matrices=False,hermitian=True)
        part_ratio = np.square(s.sum()) / np.square(s).sum()
        return part_ratio


    def compute_metrics(self,cutoff_thresh=0.95):
        zDim = self.fa_params['zDim']
        dshared = self.compute_dshared(cutoff_thresh=cutoff_thresh)
        avg_psv,psv = self.compute_psv()
        part_ratio = self.compute_part_ratio()
        metrics = {
            'psv':avg_psv,
            'ind_psv':psv,
            'dshared':dshared,
            'part_ratio':part_ratio,
            'zDim':zDim
        }
        return metrics


    def compute_earlyStop_metrics(self,cutoff_thresh=0.95):
        # compute this as the average across the early stop parameters
        zDim,dshared,part_ratio,avg_psv = [],[],[],[]
        for params in self.earlyStop_params:
            tmp_mdl = factor_analysis(model_type=self.model_type,min_var=self.min_var)
            tmp_mdl.set_params(params)
            tmp = tmp_mdl.compute_metrics(cutoff_thresh=cutoff_thresh)
            zDim.append(tmp['zDim'])
            dshared.append(tmp['dshared'])
            part_ratio.append(tmp['part_ratio'])
            avg_psv.append(tmp['psv'])
        metrics = {
            'psv':np.mean(np.array(avg_psv)),
            'dshared':np.mean(np.array(dshared)),
            'part_ratio':np.mean(np.array(part_ratio)),
            'zDim':np.mean(np.array(zDim))
        }
        return metrics


    def compute_cv_psv(self,X,zDim,rand_seed=None,n_boots=100,test_size=0.1,verbose=False,early_stop=True,return_each=False):
        # create k-fold iterator
        if verbose:
            print('Crossvalidating percent shared variance...')
        cv_folds = ms.ShuffleSplit(n_splits=n_boots,random_state=rand_seed,\
            train_size=1-test_size,test_size=test_size)

        # iterate through train/test splits
        i = 0
        train_psv,test_psv = np.zeros(n_boots),np.zeros(n_boots)
        for train_idx,test_idx in cv_folds.split(X):
            if verbose:
                print('   Bootstrap sample ',i+1,' of ',n_boots,'...')

            X_train,X_test = X[train_idx], X[test_idx]
            
            # train model
            tmp = factor_analysis(self.model_type,self.min_var)
            if early_stop:
                tmp.train(X_train,zDim,rand_seed=rand_seed,X_early_stop=X_test)
            else:
                tmp.train(X_train,zDim,rand_seed=rand_seed)

            train_psv[i] = tmp.compute_psv_heldout(X_train)
            test_psv[i] = tmp.compute_psv_heldout(X_test)

            i = i+1

        if return_each:
            return train_psv, test_psv
        else:
            return np.mean(train_psv),np.mean(test_psv)

