import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sklearn
import pickle
import xgboost


@st.cache_data
def train_model():
    data = pd.read_csv('train.csv')
    model =pickle.load(open('trained_model.pkl','rb'))
    preprocessor = pickle.load(open('process.pkl','rb'))
    X = data.drop('is_fraud',axis=1)
    y = data['is_fraud']
    X_train,X_test,y_train,y_test= sklearn.model_selection.train_test_split(X,
    y,test_size=0.3,random_state=2)
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    train_score = f1_score(y_train,tr_pred)
    test_score = f1_score(y_test,te_pred)
    return (preprocessor,model,train_score,test_score)



def main():

    
    st.sidebar.header('Dataset to use')
    page = st.sidebar.selectbox("Data Input", ['Default','User Upload'])
    
    
    
    st.title('Credit Card Fraud Predictor')
    
    with st.spinner("Unpacking the model... Please wait."):
        preprocessor,model,trainscore,testscore = train_model()
    st.write('train f1 score:', trainscore)
    st.write('test f1 score:', testscore)

    model_filename = "trained_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    if page == 'Default':
        st.header('Predicting Default Test Data')
        st.subheader('Dataset Preview')
        test = pd.read_csv('test.csv')
        test

        X = test.drop('is_fraud',axis=1)
        data = pd.DataFrame(preprocessor.fit_transform(X))
        prediction=model.predict(data)

        st.subheader('Results')
        prediction

        f1= f1_score(test['is_fraud'],prediction)
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

            testh = pd.DataFrame(preprocessor.fit_transform(data))
            prediction=model.predict(testh)

            st.subheader('Results')
            prediction

        else:
            st.write('No dataset Uploaded')
    
        





if __name__ == '__main__':
    main()
