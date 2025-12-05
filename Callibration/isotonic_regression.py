import numpy as np
from sklearn.isotonic import IsotonicRegression

def isotonic_multi(y_true, probs):
  N, C = probs.shape
  calibrated = np.zeros_like(probs)
  
  for c in range(C):
    iso = IsotonicRegression(out_of_bounds = "clip")
    calibrated[:, c] = iso.fit_transform(probs[:, c], (y_true == c))
    
  calibrated /= calibrated.sum(axis = 1, keepdims = True)
  return calibrated

# Apply Isotonic Regression
prob_iso = isotonic_multi(y_true, y_pred_proba)