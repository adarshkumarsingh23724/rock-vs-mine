import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
sonar_df=pd.read_csv('Copy of sonar data.csv',header=None)
sonar_df.head()
sonar_df.groupby(60).mean()
sonar_df.describe()
X=sonar_df.drop(columns=60,axis=1)
y=sonar_df[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression()     
model.fit(X_train, y_train)    
train_pred = model.predict(X_train)
print(accuracy_score(train_pred, y_train))  
train_pred = model.predict(X_test)
print(accuracy_score(train_pred, y_test))  

##we app---------
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Sonar Rock vs Mine Production")
input_data=st.text_input("Enter here")
if st.button('predict'):
    input_data_np_array = np.asarray(input_data.split(','),dtype=float)
    reshaped_input_data=input_data_np_array.reshape(1,-1)
    prediction=model.predict(reshaped_input_data)
    if prediction[0] == 'R':
        st.write('this is a rock')
    else:
        st.write('this is a mine')