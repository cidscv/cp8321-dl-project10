import numpy as np

def ECE_multi(y_true, probs, bins = 15):
  N, C = probs.shape
  confidences = np.max(probs, axis = 1)
  predictions = np.argmax(probs, axis = 1)
  
  accuracies = (predictions == y_true).astype(float)
  
  bin_edges = np.linspace(0, 1, bins + 1)
  ece = 0
  
  for i in range (bins):
    start, end = bin_edges[i], bin_edges[i + 1]
    mask = (confidences >= start) & (confidences < end)
    
    if mask.sum() > 0:
      avg_conf = confidences[mask].mean()
      avg_acc = accuracies[mask].mean()
      ece += np.abs(avg_conf - avg_acc) * mask.sum() / N
  
  return ece