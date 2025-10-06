import streamlit as st
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd
import pickle


# step1 : load the model.h5 file 
model = tf.keras.models.load_model('model.h5')



# step2 : load all the pickle file 
with open("ohe_geo.pkl",'rb') as file:
    lable_enc_geo=pickle.load(file)

with open('lable_enc_gen.pkl','rb') as file:
    lable_enc_gender= pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)


# start with steamlit app
st.title("Customer Churn Prediction")

# user input 
geography  = st.selectbox('Geography', lable_enc_geo.categories_[0])
Gender  = st.selectbox('Gender', lable_enc_gender.classes_)
age = st.slider('Age',18,92)
balence = st.number_input("balence")
credit_score = st.number_input("credit_score")
estimaded_salery = st.number_input("salery")
tenure = st.slider('tenure', 0,10)
num_of_product = st.slider("num_of_product",0,4)
has_cr_card = st.selectbox('has credit card',[0,1])
is_active_member = st.selectbox('is active member',[0,1])


#  prepare input data 

# Encode Gender
gender_encoded = lable_enc_gender.transform([Gender])[0]

input_data = pd.DataFrame(
    {
     "CreditScore" : [credit_score],
     "Gender": [gender_encoded],
      
     "Age"	: [age],
     "Tenure": [tenure],
     "Balance":	[balence],
     "NumOfProducts":[num_of_product],	
     "HasCrCard":[has_cr_card],	
     "IsActiveMember"	:[is_active_member],
     "EstimatedSalary"	:[estimaded_salery ]
     
    }
)




# one hot encoded 'geography'
geo_encoded = lable_enc_geo.transform([[geography]]).toarray()

geo_encoded_df= pd.DataFrame(geo_encoded,columns=lable_enc_geo.get_feature_names_out(["Geography"]))

#  combine this with input data 
input_data= pd.concat([input_data.reset_index(drop=True),
                      geo_encoded_df.reset_index(drop=True)], axis=1)


# scale input data 
input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]
if prediction_prob > 0.5 :
    st.write("the person is likely to churn")

else :
    st.write("the person will not churn ") 