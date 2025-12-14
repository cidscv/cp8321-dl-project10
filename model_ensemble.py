import numpy as np
import scipy.optimize as opt
from sklearn.metrics import log_loss, brier_score_loss

# Define evaluation function
def evaluate(name, prob):
  print(f"\n===={name}====")
  print("Log Loss: ", log_loss(y_true, probs))
  print("ECE: ", ECE_multi(y_true, probs))
 
# Evaluate models 
evaluate("Base", y_pred_proba)
evaluate("Temperature Scaled", prob_temp)
evaluate("Histogram Binned", prob_hist)
evaluate("Istonic Regressed", prob_iso)

# Weighted ensemble by minimized log loss
def loss_func(weights):
  w0, w1, w2, w3 = weights
  prob = (w0 * y_pred_proba + w1 * prob_temp + w2 * prob_hist + w3 * prob_iso
  )
  
  return log_loss(y_true, prob)

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0 , 1) * 4]
init_w = np.ones(4) / 4

result = opt.minimuze(loss_func, init_w, bounds = bounds, constraints = constraints)
best_weight = result.x

# Apply new weights
prob_weighted_ensemble = (best_weight[0] * y_pred_proba + best_weight[1] * prob_temp + best_weight[2] * prob_hist+ best_weight[3] * prob_iso)

# Select best model
models = {
  "Base": y_pred_proba, 
  "Temperature Scaled": prob_temp,
  "Histogram Binned": prob_hist,
  "Istonic Regressed": prob_iso,
  "Weighted Ensemble": prob_weighted_ensemble, 
  }

best_model = min(models, key = lambda m: log_loss(y_true, models[m]))
best_model
