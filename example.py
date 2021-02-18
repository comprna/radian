# Source: https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy

# Create the model inside the strategy's scope so that it is a 
# mirrored variable.

mirrored_strategy = MirroredStrategy()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()

# Create the input dataset.

BATCH_SIZE_PER_REPLICA = 5
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(
    global_batch_size)

# Distribute the dataset based on the strategy.

dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

# Define a custom training step

loss_object = tf.keras.losses.BinaryCrossentropy(
  from_logits=True,
  reduction=tf.keras.losses.Reduction.NONE)

def compute_loss(labels, predictions):
  per_example_loss = loss_object(labels, predictions)
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

def train_step(inputs):
  features, labels = inputs
  print("Train step: {}".format(tf.distribute.get_replica_context()))

  with tf.GradientTape() as tape:
    print("Gradient tape: {}".format(tf.distribute.get_replica_context()))
    predictions = model(features, training=True)
    loss = compute_loss(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

# Pass training step to strategy.run with the distributed data

@tf.function
def distributed_train_step(dist_inputs):
  print("Dist train step 54: {}".format(tf.distribute.get_replica_context()))
  per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
  print("Dist train step 56: {}".format(tf.distribute.get_replica_context()))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

# Iterate over the distributed dataset to run the training in a loop.

for dist_inputs in dist_dataset:
  print("For loop: {}".format(tf.distribute.get_replica_context()))
  print(distributed_train_step(dist_inputs))