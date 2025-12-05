import numpy as np

# Define histogram binning
def histogram_bin_multi(y_true, probs, bins = 10):
  N, C = probs.shape
  calibrated = np.zeros_like(probs)
  
  for c in range(C):
    prob = probs[:, c]
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(p, bin_edges) - 1
    
    bin_sums = np.bincount()
    bin_total = np.bincount()
    nonzero = bin_total != 0
    
    bin_avg = np.zeros(bins)
    bin_avg[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    calibrated[:, c] = bin_avg[bin_ids]
  
  calibrated /= calibrated.sum(axis = 1, keepdims = True)
  
# Apply histogram binning
prob_hist = histogram_bin_multi(y_true, y_pred_proba, bins = 10)