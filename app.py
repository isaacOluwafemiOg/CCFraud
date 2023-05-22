import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sklearn
import pickle


@st.cache_data
def train_model():
    test = pd.read_csv('test.csv')
    data = pd.read_csv('better_train.csv')
    
    X = data.drop('is_fraud',axis=1)
    y = data['is_fraud']
    X_train,X_test,y_train,y_test= sklearn.model_selection.train_test_split(X,
    y,test_size=0.3,random_state=2)
    return (X_test,test)

@st.cache_data
def predict_test(pdata):
    model =pickle.load(open('b_trained_model.pkl','rb'))
    pred = model.predict(pdata)
    
    return (pred)

@st.cache_data
def training_model():
    data = pd.read_csv('better_train.csv')
    model =pickle.load(open('untrained_model.pkl','rb'))
    X = data.drop('is_fraud',axis=1)
    y = data['is_fraud']
    X_train,X_test,y_train,y_test= sklearn.model_selection.train_test_split(X,
    y,test_size=0.3,random_state=2)
    model = model.fit(X_train,y_train)
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    train_score = f1_score(y_train,tr_pred)
    test_score = f1_score(y_test,te_pred)
    return (preprocessor,model,train_score,test_score)  
    
@st.cache_data
def prepare(test):
    preprocessor = pickle.load(open('process.pkl','rb'))
    X = test.drop('is_fraud',axis=1)
    data = pd.DataFrame(preprocessor.transform(X))
    
    return (data)

def main():
    

   


    st.sidebar.header('Dataset to use')
    page = st.sidebar.selectbox("Data Input", ['Default','User Upload'])
    
    
    
    st.title('Credit Card Fraud Predictor')
    
    with st.spinner("Training the model... Please wait."):
        model,trainscore,testscore = training_model()
    st.write('train score:', trainscore)
    st.write('test score:', testscore)
    
    with st.spinner("Unpacking the model... Please wait."):
        X_test,test = train_model()

    if page == 'Default':
        st.header('Predicting Default Test Data')
        st.subheader('Dataset Preview')
        
        test

        if st.button("Predict on unseen test data above"):
            with st.spinner("Preparing data... Please wait."):
                pdata = prepare(test)
                
            with st.spinner("Predicting... Please wait."):
                predic = predict_test(pdata)

            st.subheader('Results')
            predic

            f1= f1_score(test['is_fraud'],predic)
            st.subheader('F1 Score')
            f1
        st.write('''***''')


    elif page == 'User Upload':
        st.header('Predicting on Uploaded Data')
        u_data = st.file_uploader('The transactions on which you want to predict',type='csv')
        if u_data is not None:
            data = pd.read_csv(u_data,index_col = 'ID')

            st.subheader('Dataset Preview')
            data

 

        else:
            st.write('No dataset Uploaded')
    
        





if __name__ == '__main__':
    main()
