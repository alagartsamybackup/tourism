import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

st.set_page_config(page_title='Tourism Predictor', layout='wide')

@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id='alagarst/tourism-wellness-model', filename='best_model.pkl', repo_type='model')
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def prepare_features(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18):
    g = 1 if a2 == 'Male' else 0
    p = 1 if a12 == 'Yes' else 0
    c = 1 if a13 == 'Yes' else 0
    occ = {'Salaried':0,'Freelancer':1,'Self Employed':2,'Small Business':3,'Large Business':4}
    mar = {'Single':0,'Married':1,'Divorced':2}
    con = {'Company Invited':0,'Self Inquiry':1}
    pro = {'Basic':0,'Standard':1,'Deluxe':2,'Super Deluxe':3,'King':4}
    des = {'Manager':0,'Executive':1,'Senior Manager':2,'AVP':3,'VP':4}
    return np.array([[a1,con[a14],a3,a15,occ[a4],g,a8,a17,pro[a16],a10,mar[a7],a11,p,a18,c,a9,des[a6],a5]])

st.title('Tourism Package Prediction')
model = load_model()

if model:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Demographics')
        v1 = st.slider('Age', 18, 80, 35)
        v2 = st.selectbox('Gender', ['Male', 'Female'])
        v3 = st.selectbox('City Tier', [1, 2, 3])
        v4 = st.selectbox('Occupation', ['Salaried','Freelancer','Self Employed','Small Business','Large Business'])
        v5 = st.number_input('Income', 10000, 500000, 50000)
        v6 = st.selectbox('Designation', ['Manager','Executive','Senior Manager','AVP','VP'])
        v7 = st.selectbox('Marital Status', ['Single','Married','Divorced'])
    with c2:
        st.subheader('Travel')
        v8 = st.slider('People', 1, 10, 2)
        v9 = st.slider('Children', 0, 5, 0)
        v10 = st.selectbox('Hotel Stars', [3.0, 4.0, 5.0])
        v11 = st.slider('Annual Trips', 0, 20, 3)
        v12 = st.selectbox('Passport', ['Yes', 'No'])
        v13 = st.selectbox('Car', ['Yes', 'No'])
    st.subheader('Sales')
    c3, c4 = st.columns(2)
    with c3:
        v14 = st.selectbox('Contact', ['Company Invited','Self Inquiry'])
        v15 = st.slider('Pitch Duration', 5, 120, 30)
        v16 = st.selectbox('Product', ['Basic','Standard','Deluxe','Super Deluxe','King'])
    with c4:
        v17 = st.slider('Followups', 0, 10, 3)
        v18 = st.slider('Satisfaction', 1, 5, 3)
    if st.button('Predict'):
        inp = prepare_features(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18)
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        st.markdown('---')
        if pred == 1:
            st.success(f'HIGH PURCHASE LIKELIHOOD: {prob[1]:.1%}')
        else:
            st.warning(f'LOW PURCHASE LIKELIHOOD: {prob[1]:.1%}')
else:
    st.error('Model not available')
