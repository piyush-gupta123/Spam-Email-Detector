# import nltk
# from nltk import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
#Importing all the dependencies
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data using pandas
data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\NLP\\Spam Email Detector\\mail_data.csv')
# print(data.info())

# cleaning all the missing values
mail_data = data.where((pd.notnull(data)),'')
# print(mail_data.head())

# Label Encoding
#Label spam mails as 0 && ham mails as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

x = mail_data['Message']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=3)

#Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# print(x_train_features)

# Creating the model
model = LogisticRegression()
model.fit(x_train_features, y_train)

prediction = model.predict(x_test_features)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)
