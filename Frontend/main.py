import streamlit as st
import requests

# API base URL
api_url = 'http://127.0.0.1:8000'

# Set page configurations
st.set_page_config(
    page_title="Sepsis Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Sepsis Prediction Features")
st.write('- Sepsis: Positive if a patient in ICU will develop sepsis, and Negative otherwise')

# Model selection
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox('Select Model', options=['GradientBoosting', 'LogisticRegression', 'SVM', 'XGBoost'], key='model_name')

# Function to display form inputs
def show_form():
    with st.form('enter_features'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("PRG - Plasma glucose level", min_value=0, max_value=100, key='PRG')
            st.number_input("PL - Blood Work Result-1 (mu U/ml)", min_value=0, max_value=500, key='PL')
            st.number_input("PR - Blood Pressure (mm Hg)", min_value=0, max_value=500, key='PR')
        with col2:        
            st.number_input("SK - Blood Work Result-2 (mm)", min_value=0, max_value=500, key='SK')
            st.number_input("TS - Blood Work Result-3 (mu U/ml)", min_value=0, max_value=500, key='TS')
            st.number_input('M11 - Body mass index (BMI)', min_value=0, max_value=200, key='M11')
        with col3:
            st.number_input('BD2 - Blood Work Result-4 (mu U/ml)', min_value=0, max_value=100, key='BD2')
            st.number_input("Age - Patient's age (years)", min_value=0, max_value=150, key='Age')
            st.number_input("Insurance - Valid insurance card (0 or 1)", min_value=0, max_value=1, key='Insurance')
        st.form_submit_button('Predict Sepsis Status', on_click=make_prediction)

# Function to make prediction
def make_prediction():
    data = {
        'PRG': st.session_state['PRG'],
        'PL': st.session_state['PL'],
        'PR': st.session_state['PR'],
        'SK': st.session_state['SK'],
        'TS': st.session_state['TS'],
        'M11': st.session_state['M11'],
        'BD2': st.session_state['BD2'],
        'Age': st.session_state['Age'],
        'Insurance': st.session_state['Insurance']
    }
    
    # Make the API request
    response = requests.post(f'{api_url}/predict/{model_name}', json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        st.session_state['prediction'] = response_data['prediction']
        st.session_state['probability'] = response_data.get('probability', 'N/A')
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Display the form and results
if __name__ == "__main__":  
    show_form()
    
    if 'prediction' in st.session_state:
        final_prediction = st.session_state['prediction']
        prob = st.session_state.get('probability', 'N/A')

        col1, col2 = st.columns(2)
        with col1:
            if final_prediction == "Positive":
                st.write("### Sepsis test will be :red[Positive]\nPatient is likely to develop Sepsis")
            else: 
                st.write(f'### Sepsis test will be :green[Negative]\nPatient is not likely to develop Sepsis')     
        with col2:
            st.subheader('@ What Probability?')
            if isinstance(prob, list) and len(prob) == 2:
                st.write(f'#### :red[{round(prob[1] * 100, 2)}%] chance of Patient developing Sepsis.')
            else:
                st.write(f'#### Probability: {prob}')
    else:
        st.write('### Prediction will be shown here after submitting the form.')
