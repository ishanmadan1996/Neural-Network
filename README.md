# Deep Neural Networks

Creating a neural network model using Google's tensorflow open source library, and then training it to perform sentiment classification on given data.

## Usage

Clone the repository:

```
git clone https://github.com/ishanmadan1996/Neural-Network.git
```

Install the pre-requisite libraries:

```
pip install requirements.txt
```
Download the 140 sentiment dataset from 
```

http://help.sentiment140.com/for-students/

```

The next step is to create a lexicon and preprocesses training and testing data using the given data which can be found in the 140 sentiment dataset. The script given below is used for this purpose. After executing the script, the lexicon can be stored in a pkl file (lexicon-2500.pickle) which can be later used by our neural network.
```
preproccessing_Sentiment_140_dataset.py
```

The second step is to run the script given below, in order to train the featureset which we obtained from executing 'preproccessing_Sentiment_140_dataset.py'. There is function in this script called use_neural_netowrk, which can be used to test the neural network, by providing hard coded sentences to it, so that it can determine their sentiment.
```
sentiment_network_140.py
```
The 'lexicon-1500.pickle' file is the lexicon obtained from the first 1500 samples of 'train_set.csv'. 'model.pickle' is a pre-trained model using this lexicon.

##Output
The output consists of the accuracy of the model in correcttly classifying the sentiment of the tweet. As we can see the accuracy is a bit low, which can be improved by taking more number of samples from dataset.

```
Epoch 10 completed out of 10 loss: 1565.3750583734563
Tested 357 samples.
Accuracy: 0.63305324

Negative: No, I hate her
Positive: This was the best store i've ever seen.

```

## Built With

* [Python](https://www.python.org/doc/) - The scripting language used

## Authors

* **Ishan Madan** - (https://github.com/ishanmadan1996)

