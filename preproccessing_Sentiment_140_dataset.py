import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import csv
from collections import Counter
csv_database = create_engine('sqlite:///csv_database.db') #database to store big files
chunksize = 200000
lemmatizer = WordNetLemmatizer()

'''
140 sentiment dataset 

polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

#preprocess the given dataset by converting positive polarities to [1,0] and negative ones to [0,1]
def init_process(fin,fout):
    i = 0
    j = 1
    for df in pd.read_csv(fin, chunksize=chunksize, iterator=True,encoding='latin-1'): #reading the input data in chunks (as data is very large).Storing each chunk in sqlite db. To only store chunksize amount of data at a sinle run , iterator= false.
        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
        df.index += j
        i+=1
        df.to_sql('tweetstest', csv_database, if_exists='append') #save chunk of data in sqlite db. Avoid using uppercase letters for table name
        j = df.index[-1] + 1
    df = pd.read_sql_query('SELECT *FROM tweetstest', csv_database) #retrieve the whole database from sqlite. You can use complex sql queries to filter your data and retrieve it as per your dataset.
    cols = [2,3,4,5]
    df.drop(df.columns[cols], axis=1)
    try:
        for index, row in df.iterrows():
           
            #choosing first column (polarity) from csv data
            if str(row[df.columns[1]]) == '0':
                initial_polarity = '[0,1]' #negative sentiment  [positive,negative]
                tweet = str(row[df.columns[6]])#choosing last column (tweet)
                with open(fout,'a',encoding='latin-1', newline='') as f:
                	writer = csv.writer(f)
                	rows=zip([initial_polarity],[tweet])
                	for row in rows:
                		print (row)
                		writer.writerow(row)
                		f.close()
            elif str(row[df.columns[1]]) == '4':
                initial_polarity = '[1,0]' #positive sentiment
                tweet = str(row[df.columns[6]])#choosing last column (tweet)
                with open(fout,'a',encoding='latin-1', newline='') as f:
                	writer = csv.writer(f)
                	rows=zip([initial_polarity],[tweet])
                	for row in rows:
                		print (row)
                		writer.writerow(row)
                		f.close()

            
    except Exception as e:
        raise e

init_process('training1600000processed_shuffled.csv','train_set.csv')
init_process('testdata.manual.2009.06.14.csv','test_set.csv')

#create lexicon of words from the training data. We will use this lexicon to vectorise each tweet. LenOfVector(every tweet) = len(lexicon)
def create_lexicon(fin):
	lexicon = []
	df = pd.read_csv(fin,error_bad_lines=False,encoding='latin-1')
	try:
		counter = 1
		content = ''
		for i in range(1500): #reading first 2500 tweets from training set
			counter += 1
			tweet = str(df.at[i,df.columns[1]])
			print(str(tweet))
			content += ' '+tweet
			words = word_tokenize(content)
			words = [lemmatizer.lemmatize(i) for i in words]
			lexicon = list(set(lexicon + words))
			print(counter,len(lexicon))
	except Exception as e:
		print(str(e))
	
	with open('lexicon-1500.pickle','wb') as f: #lexicon of words collected from first 1500 lines/tweets/samples.No of words = 5472
		pickle.dump(lexicon,f)


create_lexicon('train_set.csv')

#vectorise test_data using lexicon of first 'x' number of words
def convert_to_vec(fin,fout,lexicon_pickle):
	with open(lexicon_pickle,'rb') as f:
		lexicon = pickle.load(f)
	
	df = pd.read_csv(fin,encoding='latin-1')
	counter = 0
	for row in range(358): #length of test_set file
		counter +=1
		label = str(df.at[row,df.columns[0]])
		tweet = str(df.at[row,df.columns[1]])
		current_words = word_tokenize(tweet.lower()) #tokenize the tweet into bag of words
		current_words = [lemmatizer.lemmatize(i) for i in current_words] #lemmatize each word in the bag

		features = np.zeros(len(lexicon)) #create a numpy array/vector of lists(bag of words)

		for word in current_words:
			if word.lower() in lexicon:
				index_value = lexicon.index(word.lower())
				# OR DO +=1, test both
				features[index_value] += 1

		features = list(features)
		with open(fout,'a',encoding='latin-1', newline='') as f:
			writer = csv.writer(f)
			rows=zip([label],[features])
			for row in rows:
				print (row)
				writer.writerow(row)
				f.close() 


	print(counter)

convert_to_vec('test_set.csv','processed-test-set.csv','lexicon-1500.pickle')

#shuffle the input dataset
def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines=False,encoding='latin-1')
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('training1600000processed_shuffled.csv', index=False)
	
shuffle_data('training1600000processed.csv')


def create_test_data_pickle(fin):

	feature_sets = []
	labels = []
	counter = 0
	df = pd.read_csv(fin)
	for rwo in range(620):
		try:
			features = list(eval(str(df.at[row,df.columns[1]]))) #tweets
			label = list(eval(str(df.at[row,df.columns[0]]))) #polarity

			feature_sets.append(features)
			labels.append(label)
			counter += 1
		except:
			pass
	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')

# with open('lexicon-2500.pickle','rb') as f:
# 	lexicon = pickle.load(f)
# 	print(len(lexicon)) 

# df = pd.read_csv('train_set.csv',encoding='latin-1')
# print(Counter(df.at['[0,1]',df.columns[0]]))