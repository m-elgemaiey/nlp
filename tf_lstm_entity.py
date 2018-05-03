import spacy
import json
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

entity_set = set()

with open('nlu_training.json') as json_file:
    json_data = json.load(json_file)
    examples = json_data['rasa_nlu_data']['common_examples']
    for example in examples:
        entities = example['entities']
        if entities:
            for entity in entities:
                entity_set.add(entity['entity'])

#print(entity_set)

entity_dict = {value:i+1 for i, value in enumerate(entity_set)}
entity_rev_dict = {i+1:value for i, value in enumerate(entity_set)}
entity_dict['None'] = 0
entity_rev_dict[0] = 'None'
entity_set.add('None')

print(entity_dict)

time_steps=10
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=300
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=len(entity_set)
#size of batch
batch_size=len(examples)

print("Batch size: {0}, Steps: {1}, # Classes: {2}, ".format( batch_size, time_steps, n_classes))
# spacy stuff

print("Loading spacy model")
nlp = spacy.load('en_core_web_md')

def genTrainingData(examples):
    num_sentences = len(examples)
    input_x = np.zeros((num_sentences, time_steps, n_input))
    input_y = np.zeros((num_sentences, time_steps))

    for i in range(num_sentences):
        example = examples[i]
        doc = nlp(example['text'])
        j = 0
        for token in doc:
            input_x[i, j] = token.vector
            entities = example['entities']
            if entities:
                for entity in entities:
                    if token.idx >= entity['start'] and token.idx < entity['end']:
                        input_y[i, j] = entity_dict[entity['entity']]
                        break
            j += 1
            if j == time_steps:
                break
            
    return input_x, input_y


#test_sents = ['good afternoon', 'show me an indian resturant', 'afternoon good']
#test_intents = ['greet', 'restaurant_search', 'greet']
'''
test_sents = ['hi customer', 'show me sales by region', 'afternoon good', 'good morning customer']
test_intents = ['greet', 'show', 'greet', 'greet']

test_x, test_y = genTrainingData(test_sents, test_intents)
'''
#print('input_x', input_x)
#print('input_y', input_y)
#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("int32",[None,time_steps])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
outputs = tf.reshape(outputs, [-1, num_units])

out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))
logits = tf.nn.xw_plus_b(outputs, out_weights, out_bias)
logits = tf.reshape(logits, [-1, time_steps, n_classes])
loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            y,
            tf.ones([batch_size, time_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
loss = tf.reduce_sum(loss)
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, n_classes]))
prediction = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
#model evaluation
correct_prediction = tf.equal(prediction, tf.reshape(y, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_x, train_y = genTrainingData(examples)
print(examples[51]['text'])
print(train_y[51])

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<800:
        sess.run(opt, feed_dict={x: train_x, y: train_y})

        if iter %10==0:
            acc, los = sess.run([accuracy, loss],feed_dict={x:train_x,y:train_y})
            print(iter, acc, los, sep = '\t')

        iter=iter+1

    pred = sess.run(prediction, feed_dict={x: train_x, y: train_y})
    shaped_pred = np.reshape(pred, (batch_size, time_steps))
    #print("Prediction:", shaped_pred)
    #print(len(shaped_pred))
    
    for i in range(len(examples)):
        print(examples[i]['text'])
        print([entity_rev_dict[train_y[i, j]] for j in range(time_steps)])
        print([entity_rev_dict[shaped_pred[i, j]] for j in range(time_steps)])
