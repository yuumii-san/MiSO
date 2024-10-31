import numpy as np
        
def align_loadings(L1, L2, usable_chan1, usable_chan2, commonChans=None):
    """
    Align two FA loading using   

    Parameters:
        L1 (numpy.ndarray): FA Loading matrix for session 1 (num_usable_chan x latent dimensionality).
        L2 (numpy.ndarray): FA Loading matrix for session 2 (num_usable_chan x latent dimensionality).
        usable_chan1 (list): List of usable channels for session 1.
        usable_chan2 (list): List of usable channels for session 2.
        commonChans (list): List of common usable channels.

    Returns:
        numpy.ndarray: Aligned FA Loading for session 2.
    """
    if commonChans is None:
        commonChans = set(usable_chan1).intersection(usable_chan2)
    
    commonChansBool1 = [chan in commonChans for chan in usable_chan1]
    commonChansBool2 = [chan in commonChans for chan in usable_chan2]
    
    # Aligned loading (based on Ls)
    L1L2 = np.matmul(L1[commonChansBool1,:].T, L2[commonChansBool2,:])
    U, S, VT = np.linalg.svd(L1L2)
    O = np.matmul(U, VT)
    LO = np.matmul(L2, O.T)
    return LO