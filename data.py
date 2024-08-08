import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title='Data Page',
    page_icon='🗃',
    layout='wide'
)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

@st.cache_data
def load_data0():
    train_data = pd.read_csv('Dataset/Train.csv')
    return train_data

@st.cache_data
def load_data():
    test_data = pd.read_csv('Dataset/Test.csv')
    return test_data



# Function to get all columns from a specific table
st.cache_resource()
def get_all_column():
    train_data = pd.read_csv("Dataset/Test.csv")
    test_data = pd.read_csv('Dataset/Train.csv')
    
    st.session_state['train_data'], st.session_state['test_data']
# Initialize the dataframe in session state if not already there
if 'train_data' not in st.session_state and 'test_data' :
    get_all_column()
    
def select_data():
    st.subheader('ETA PREDICTION  APP')
    col1, col2 = st.columns(2)
    with col1:
        # Dropdown select box
        selection = st.selectbox('Select Columns of Data', options=['All columns', 'Numerical Columns', 'Categorical Columns'], on_change=get_all_column)
    with col2:
        pass
    # Filter the DataFrame based on the selection
    if selection == 'Numerical Columns':
        train_data_to_display = st.session_state['train_data'].select_dtypes(include=['number'])
        test_data_to_display = st.session_state['test_data'].select_dtypes(include=['number'])
        
    elif selection == 'Categorical Columns':
        train_data_to_display = st.session_state['train_data'].select_dtypes(include=['object', 'category'])
        test_data_to_display= st.session_state['test_data'].select_dtypes(include=['object', 'category'])
            
    else:
        train_data_to_display = st.session_state['train_data']
        test_data_to_display = st.session_state['test_data']
        
    # Display the DataFrame
    
    if st.checkbox('train_data'):
        st.subheader(f'**{selection}**')
        st.write(train_data_to_display)
    
    if st.checkbox('test_data'):    
        st.subheader(f'**{selection}**')
        st.write(test_data_to_display)

    
        
    
def data_description():
    st.subheader('Descriptive statistics')
    
    col1, col2 = st.columns(2)
    with col1:
        select_df = st.selectbox('Select DataFrame', options=['All Columns', 'Numerical Columns', 'Categorical Columns'], on_change=get_all_column)
    with col2:
        pass
    if select_df == 'All Columns':
        Test_df = st.session_state['test_data'].describe(include=['number', 'object', 'category']).T
        Train_df = st.session_state['train_data'].describe(include=['number', 'object', 'category']).T
        
    if select_df == 'Numerical Columns':
        Test_df = st.session_state['data'].describe(include=['number']).T
        Train_df = st.session_state['data0'].describe(include=['number']).T
        
    if select_df == 'Categorical Columns':
        Test_df = st.session_state['data'].describe(include=['object', 'category']).T
        Train_df = st.session_state['data0'].describe(include=['object', 'category']).T
        
    
    if st.checkbox('Show train data descriptive statistics'):
        st.subheader(f'**{select_df}**')
        st.write(Train_df)
    
    

   
        
def data_dict():
    st.subheader('Data Dictionary')

    with st.expander('Click to see data dictionary'):
        st.write("""
            - date
            - dewpoint_2m_temperature
            - maximum_2m_air_temperature
            - mean_2m_air_temperature
            - mean_sea_level_pressure
            - minimum_2m_air_temperature
            - surface_pressure
            - total_precipitation
            - u_component_of_wind_10m
            - v_component_of_wind_10m
        """)
        
    
if st.session_state['authentication_status']:
    authenticator.logout(location='sidebar')
    col1, col2 = st.columns(2)
    with col1:
        pass
    with col2:
        st.write('### :rainbow-background[Telco Customer Churn Data ]🗃')
        st.selectbox('Select DataFrame/Descriptive statistics', options=['Data', 'Statistics'], key='selected_dataframe')
    if st.session_state['selected_dataframe'] == 'Data':
            select_data() 
    elif st.session_state['selected_dataframe'] == 'Statistics':
            data_description()
            
    data_dict()

else:
    st.info('Login to gain access to the app')

            