import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle
import re


def preprocess(tweet):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = r'@[^\s]+'
    # setting stopwords manually to avoid downloading the package again and again
    stopword = {"won't", 'y', 'over', 'them', 'can', "couldn't", 'didn', "shan't", 'hasn', 'too', 'ain', 'other', 'off', "doesn't", 'd', 'which', 'herself', 'these', 'you', 't', 'ourselves', 'the', 'between', 'i', 'yourself', 'do', 'than', 'because', 'shan', 'here', 'shouldn', 'at', 'doing', 'such', "hadn't", 'while', "you'll", 'down', 'an', 'until', 'isn', 'during', 'theirs', 'all', 'him', 'aren', 'against', 'yourselves', 'just', 'own', "should've", "aren't", "haven't", "hasn't", 'not', 'is', 'mightn', 'your', 'am', 'now', 'their', 'be', 'to', 'don', "you're", "you'd", 'there', 'what', 'whom', 'into', 's', 'up', 'by', 'any', 'yours', 'myself', 'was', 'but', 'if', 'once', 'some', "weren't", "you've", 'does', 'he', 'more', 'further', 'it', 'hadn', 'under', "mightn't", 've', 'needn', 'of', 'below', 're', 'its', 'and', 'has', "it's", 'ma', 'with', 'this', "that'll", 'very', 'same', 'most', 'doesn', 'itself', 'out', 'she', 'a', 'as', 'themselves', 'his', "isn't", "she's", "mustn't", "don't", 'are', 'did', 'how', 'nor', 'only', 'been', 'who', 'that', 'me', 'o', 'through', 'before', 'each', "needn't", 'having', 'then', 'wouldn', 'ours', 'about', 'above', 'for', "wouldn't", 'our', 'weren', 'my', 'had', 'no', 'should', 'in', 'have', 'after', 'so', 'where', 'being', 'both', "didn't", 'when', 'few', 'why', 'they', "shouldn't", 'haven', 'm', 'hers', 'wasn', "wasn't", 'couldn', 'her', 'mustn', 'himself', 'were', 'will', 'again', 'or', 'those', 'we', 'll', 'on', 'from', 'won'}
    #stopword = set(stopwords.words('english'))

  # Lower Casing
    tweet = tweet.lower()
    tweet=tweet[1:]
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet) 
    #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #tokenizing words
    tokens = word_tokenize(tweet)
    #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
      if len(w)>1:
        word = wordLemm.lemmatize(w)
        finalwords.append(word)
    return ' '.join(finalwords)

def load_models(model):
    # Load the vectoriser.
    file = open('models/vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    if model == "LR": # Load the LR Model.
        file = open('models/logisticRegression.pickle', 'rb')
        model = pickle.load(file)
        file.close()
    elif model == "SVM": # Load the SVM Model.
        file = open('models/SVM.pickle', 'rb')
        model = pickle.load(file)
        file.close()
    elif model == "RF": # Load the RF Model.
        file = open('models/RandomForest.pickle', 'rb')
        model = pickle.load(file)
        file.close()
    elif model == "BNB": # Load the BNB Model.
        file = open('models/NaivesBayes.pickle', 'rb')
        model = pickle.load(file)
        file.close()
    return vectoriser, model

def predict(vectoriser, model, text):
    # Predict the sentiment
    processes_text=[preprocess(sen) for sen in text]
    textdata = vectoriser.transform(processes_text)
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

def predict_sentiment(vectoriser,models,text):
    # preprocess the text
    processes_text = [preprocess(sen) for sen in text]
    textdata = vectoriser.transform(processes_text)
    data = []
    for model in models:
        if model == "LR":
            vectoriser, lr = load_models(model)
            sentiment = lr.predict(textdata)
        elif model == "BNB":
            vectoriser, nb = load_models(model)
            sentiment = nb.predict(textdata)
        elif model == "SVM":
            vectoriser, svc = load_models(model)
            sentiment = svc.predict(textdata)
        for pred in sentiment:
            data.append(pred)
    df = pd.DataFrame()
    df['model'],df['sentiment'] = models,data
 
    df = df.replace([0,1], ["Negative","Positive"])
    return df