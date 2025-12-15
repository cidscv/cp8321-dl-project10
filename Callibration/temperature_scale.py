import tensorflow as tf

# Define Temperature Scaler
class TemperatureScaler(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.temperature = self.add_weight(name="temp", shape = (), initializer = tf.keras.initializers.Constant(1.0), trainable = True)

  def call(self, logits):
    return logits / self.temperature

# Logits converter
eps = 1e-8
logits = tf.math.log(y_pred_proba + eps)

# Define Temperature Scale Fitter
def fit_temperature_multi(logits, y_true, lr = 0.01, epochs = 200):
  ts = TemperatureScaler()

  _ = ts(logits[:1])

  optimize = tf.keras.optimizers.Adam(learning_rate = lr)
  y_true_tensor = tf.convert_to_tensor(y_true, dtype = tf.int32)

  for e in range(epochs):
    with tf.GradientTape() as tape:
      scaled_logits = ts(logits)
      loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels = y_true_tensor,
          logits = scaled_logits
        )
      )

    grad = tape.gradient(loss, ts.trainable_variables)
    optimize.apply_gradients(zip(grad, ts.trainable_variables))

  return ts.temperature.numpy()

# Fit Scaler to Model
temperature = fit_temperature_multi(logits, y_true)

# Apply Temperature
logits_scaled = logits / temperature
prob_temp = tf.nn.softmax(logits_scaled).numpy()
