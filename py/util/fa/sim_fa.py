import numpy as np

class sim_fa:

    def __init__(self,obs_dim,latent_dim,model_type='fa',rand_seed=None):
        self.model_type = model_type
        self.xDim = obs_dim
        self.zDim = latent_dim

        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        # generate model parameters
        mu = np.random.randn(obs_dim)
        L = np.random.randn(obs_dim,latent_dim)
        
        if model_type.lower()=='ppca':
            Ph = 10 * np.ones(obs_dim)
        elif model_type.lower()=='fa':
            Ph = np.linspace(1,obs_dim,obs_dim)
        else:
            raise ValueError('model_type should be "fa" or "ppca"')

        # store model parameters in dict
        fa_params = {'mu':mu,'L':L,'Ph':Ph,'zDim':latent_dim}
        self.fa_params = fa_params


    def sim_data(self,N,rand_seed=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        Ph = self.fa_params['Ph'].reshape(self.xDim,1)
        L = self.fa_params['L']
        mu = self.fa_params['mu'].reshape(self.xDim,1)

        # generate data
        z = np.random.randn(self.zDim,N)
        ns = np.random.randn(self.xDim,N) * np.sqrt(Ph)
        X = (L.dot(z) + ns) + mu

        return X.T


    def get_params(self):
        return self.fa_params


    def set_params(self,fa_params):
        self.fa_params = fa_params

