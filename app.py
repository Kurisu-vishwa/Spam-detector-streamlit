
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
st.header("Welcome! To spam identifier")
message = st.text_input("Enter your text here")


df= pd.read_csv('email_spam.csv',encoding='latin-1')
print(df.head())
df['v1']=df['v1'].map(({'ham':0,'spam':1}))
y = df['v1']
x = df['v2']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)
Pipe = Pipeline([('tfid', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000,random_state=42))
    ])
Pipe.fit(x_train, y_train)

y_predict = Pipe.predict(x_test)

def run(message):
    ipmessage=[message]
    result = Pipe.predict(ipmessage)[0]
    proba = Pipe.predict_proba(ipmessage)[:,1][0]
    label ="Spam" if result==1 else "Not Spam"
    st.write("Message: ", ipmessage[0])
    st.write("Prediction: ",label)
    st.write("Probability: ",proba)

if st.button("Validate"):
    run(message)
    
