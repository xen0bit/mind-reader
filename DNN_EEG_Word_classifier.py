from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://raw.githubusercontent.com/xen0bit/mind-reader/main/training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['TP9','AF7','AF8','TP10','Right AUX','wordClass']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['the', 'firm', 'time', 'out', 'such', 'did', 'bear', 'involved', 'you', 'man', 'shuddered', 'wanted,', 'they', 'with', 'son', 'as', 'over', 'but', 'thank', 'Mr.', 'Dursley', 'He', 'for', 'it', 'perfectly', 'people', 'Grunnings,', 'so', 
'amount', 'too,', 'greatest', 'hardly', 'any', 'nonsense.', 'much', 'good-for-nothing', 'mysterious,', 'years;', 'sister,', 'blonde', 'The', 'Drive,', 'unDursleyish', 'finer', 'craning', 'strange', 'found', 'in', 'Dursleys', 'if', 'discover', 'pretended', 'to', 'fear', 'and', 'no', 'neck,', 'arrived', 'made', 'proud', 'anywhere.', 'what', 'fences,', "hadn't", 'beefy', 'They', 'director', 'garden', 'Privet', 'normal,', 'drills.', 'Potter', 'her', 'a', 'street.', 'seen', 
'small', 'be.', 'on', 'spent', 'fact,', 'somebody', 'say', 'that', 'nearly', 'useful', 'had', 'sister', 'could', 'be', 'Potters.', 'him.', 'she', 'also', 'very', 'which', 'he', 'of', 'never', 'neighbors.', 'it.', 'several', 'Dudley', 'just', 'opinion', 'usual', 'was', 'there', 'neighbors', 'husband', 'came', 'Mrs.', 'have', 'thin', 'even', 'four,', 'about', 'were', 'Dursley,', 'everything', 's', 'much.', 'possible', 'expect', 'twice', 'their', 'secret,', "you'd", 'anyone', 'number', 'hold', 'or', 'boy', 'although', 'big,', 'would', 'met', 'anything', 'think', 'Potters', 'knew', "didn't", 'mustache.', 'last', 'because', 'spying', 'son,', 'large', "Dursley's", 'called']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    shuffle=True, 
    shuffle_buffer_size=10000,
    label_name=label_name,
    num_epochs=1)


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print(features[:5])



model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(len(class_names))
])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)



optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 1000

for epoch in tqdm(range(num_epochs)):
  #print('Epoch: ' + str(epoch))
  epoch_loss_avg = tf.keras.metrics.Mean()
  #print('Setting Accuracy')
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  #print(train_dataset)
  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  if epoch % 25 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),epoch_accuracy.result()))
    model.save('./model/eegwords.h5')


# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# fig.suptitle('Training Metrics')

# axes[0].set_ylabel("Loss", fontsize=14)
# axes[0].plot(train_loss_results)

# axes[1].set_ylabel("Accuracy", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)
# axes[1].plot(train_accuracy_results)
test_url = "https://raw.githubusercontent.com/xen0bit/mind-reader/main/training.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='wordClass',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    #print(logits)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    #print(prediction)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



