import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd 

lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2 #(polarity , positive or negative)

batch_size = 32 #no of tweets taken as input at one time by the model
total_batches = int(1600000/batch_size)
hm_epochs = 10 #no of epochs

x = tf.placeholder('float')  #input data, tweets. You can specify shape(length,width) of the input matrix
y = tf.placeholder('float') #label, poalrity of tweets

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([5472, n_nodes_hl1])), #input,number of nodes. Creates random weights.Weights are a tf var, with shape given in random_normal funciton
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))} #biases are added to the (input data)*weights. 

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

#the actual computational model/graph
def neural_network_model(data):
    # (inputdata * weights + biases) - actual model/neural network, given below

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1) #activation func , rectified linear

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output  #returns one hot vector/array classification of the polarity of the tweet


saver = tf.train.Saver()
tf_log = 'tf.log' #track current epoch number

#training the neural network, by using training dataset
def train_neural_network(x):
    prediction = neural_network_model(x)  #build model and save in var prediction
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess: #initialise session and start the computation graph/model
        sess.run(tf.global_variables_initializer()) #initilalise all tf models/variables
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch) #logging the epoch number when we stopped executing
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,".\model.ckpt") #restore trained model of previous epcoh
            epoch_loss = 1
            with open('lexicon-1500.pickle','rb') as f:
                lexicon = pickle.load(f)   
            df = pd.read_csv('train_set.csv',encoding='latin-1')#open training set data containing polarity,tweet(non vectorised form)
           
            batch_x = [] #input tweet
            batch_y = [] #polarity of one tweet
            batches_run = 0
            for row in range(6200): #training 6200 samples/tweets from training set
                if type(df.at[row,df.columns[0]])==str:
                    label = str(df.at[row,df.columns[0]]) #store polarity of tweet
                else:
                     pass
                if type(df.at[row,df.columns[1]])==str:
                    tweet = str(df.at[row,df.columns[1]]) #store tweet
                else:
                     pass
                current_words = word_tokenize(tweet.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]

                features = np.zeros(5472) #creating a vector of len(lexicon) for each input tweet

                #vectorise the tweeet
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        # OR DO +=1, test both
                        features[index_value] += 1
                line_x = list(features)
                line_y = eval(label)
                batch_x.append(line_x)
                batch_y.append(line_y)
                if len(batch_x) >= batch_size: #checking cost for one batch of samples
                    _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                              y: np.array(batch_y)}) #calculates the loss, and feeds the input and output data to x and y respectively
                    epoch_loss += c
                    batch_x = []
                    batch_y = []
                    batches_run +=1
                    print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            saver.save(sess, ".\model.ckpt") #save trained model of current epoch
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f: #write epoch number to log file
                f.write(str(epoch)+'\n') 
            epoch +=1

# train_neural_network(x)



#to test the neural network with testing set
def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs): #restore the saved and already trained model
            try:
                saver.restore(sess,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) #argmax returns index of max value. We use it to compare predicted vs actual o/p
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) #tf.cast() changes datatype of var.
        feature_sets = []
        labels = []
        counter = 0
        df = pd.read_csv('processed-test-set.csv')
        for row in range(358): #saving 
            try:
                features = list(eval(str(df.at[row,df.columns[1]])))
                label = list(eval(str(df.at[row,df.columns[0]])))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
        print('Tested',counter,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


# test_neural_network()

#use neural network to test sentiment of hardcoded input string
def use_neural_network(input_data):
    prediction = neural_network_model(x) #building the model and storing in a variable
    with open('lexicon-1500.pickle','rb') as f:
        lexicon = pickle.load(f) #load pkl file into variable
        
    with tf.Session() as sess: #start the tf session
        sess.run(tf.global_variables_initializer()) #initiliase all tf variables
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        #vectorise the tweet using lexicon
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax gives index of larger number: 0(index of positive class)
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        print(prediction.eval(feed_dict={x:[features]}),1)
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

use_neural_network("No, I hate her")
use_neural_network("This was the best store i've ever seen.")