"""Proposal scores; predictions never replace measured simulation outcomes."""
import numpy as np
from scipy.special import ndtr

def scores(mean,std,probability,incumbent):
    mean=np.asarray(mean);std=np.maximum(np.asarray(std),1e-9)
    delta=mean-incumbent;z=delta/std
    # Gaussian approximation to improvement, using ensemble spread as a heuristic.
    # It is not a calibrated probability or a confidence interval on performance.
    expected=np.maximum(0,delta*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi))
    feasibility=.1+.9*np.asarray(probability)
    performance=np.maximum(mean,0)*feasibility
    gain=expected*feasibility
    assert np.isfinite(performance).all() and np.isfinite(gain).all()
    return performance,gain

if __name__=='__main__':
    performance,gain=scores([100.,200.],[10.,10.],[.8,.8],225.)
    assert performance[1]>performance[0] and gain[1]>gain[0]
    print('Equal uncertainty and feasibility: a higher predicted mean receives higher performance and improvement scores.')
