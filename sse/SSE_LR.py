
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import math

frames = pd.read_csv('finalsse.csv', sep = ',')
frames['UsedTime'] /= 1000
frames['Score'] /= 10
frames
frames.describe()

my_feature = frames[['UsedTime']]
feature_columns = [tf.feature_column.numeric_column('UsedTime')]

targets = frames['Score']

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

lr = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=100)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

_ = lr.train(
    input_fn = lambda: my_input_fn(my_feature, targets),
    steps=10
)

prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
predictions = lr.predict(input_fn = prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

plt.figure(figsize=(8, 6))
plt.xlabel('UsedTime')
plt.ylabel('Score')

sample = frames.sample(n=90)
X_0 = sample['UsedTime'].min()
#print(X_0)
X_1 = sample['UsedTime'].max()
#print(X_1)
#lr.get_variable_names()
weight = lr.get_variable_value('linear/linear_model/UsedTime/weights')[0]
bias = lr.get_variable_value('linear/linear_model/bias_weights')

y_0 = weight * X_0 + bias
y_1 = weight * X_1 + bias
plt.plot([X_0, X_1], [y_0, y_1], c='r')

plt.scatter(sample['UsedTime'], sample['Score'])

plt.show()
# =====================================================================

