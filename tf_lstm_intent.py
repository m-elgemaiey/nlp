import spacy
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import utils as utils

sentences, intents, intent_set = utils.load_intents('data/nlu_training.json')

intent_dict = {value:i for i, value in enumerate(intent_set)}
intent__rev_dict = {i:value for i, value in enumerate(intent_set)}

time_steps=7
#hidden LSTM units
num_units=128
#Word vector size
n_input=300
#learning rate for adam
learning_rate=0.001
# number of intents.
n_classes=len(intent_set)
#size of batch
batch_size=len(intents)

print("# Classes: {0}, Batch size: {1}".format(n_classes, batch_size))
# spacy stuff
print('Loading spacy model ...')
nlp = spacy.load('en_core_web_md')

def genTrainingData(sentences, intents):
    num_sentences = len(sentences)
    input_x = np.zeros((num_sentences, time_steps, n_input))
    input_y = np.zeros((num_sentences, n_classes))

    for i in range(num_sentences):
        doc = nlp(sentences[i])
        j = 0
        for token in doc:
            input_x[i, j] = token.vector
            j += 1
            if j == time_steps:
                break
        input_y[i, intent_dict[intents[i]]] = 1

    return input_x, input_y

train_x, train_y = genTrainingData(sentences, intents)

#test_sents = ['good afternoon', 'show me an indian resturant', 'afternoon good']
#test_intents = ['greet', 'restaurant_search', 'greet']
test_sents = ['hi customer', 'show me sales by region', 'afternoon good', 'good morning customer']
test_intents = ['greet', 'show', 'greet', 'greet']

test_x, test_y = genTrainingData(test_sents, test_intents)

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.nn.softmax(tf.matmul(outputs[-1],out_weights)+out_bias)

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('iter', 'accuracy', 'loss', sep = '\t')
    iter=1
    while iter<=200:
        sess.run(opt, feed_dict={x: train_x, y: train_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:train_x,y:train_y})
            los=sess.run(loss,feed_dict={x:train_x,y:train_y})
            print(iter, acc, los, sep = '\t')

        iter=iter+1
        
    acc, pred = sess.run([accuracy, prediction], feed_dict={x: test_x, y: test_y})
    print('Expect\tActual\tSentence')
    for i in range(len(test_sents)):
        print(test_intents[i], intent__rev_dict[np.argmax(pred[i])], test_sents[i],sep='\t')
    print("Testing Accuracy:", acc)
