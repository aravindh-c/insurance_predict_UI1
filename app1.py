import streamlit as st
import base64
import numpy as np
import pandas as pd
import joblib

import streamlit as st
import base64


def add_bg_from_local(image_file):
    # Read and encode the image file
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    # Apply the background image and styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Make the content more readable with a semi-transparent white background */
        .stMarkdown, .stHeader, div[data-testid="stVerticalBlock"] > div {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 0; /* Remove default margin */
        }}

        /* Remove padding from the main container */
        .stContainer {{
            padding: 0; /* Adjust as necessary */
        }}

        /* Remove padding from specific vertical blocks */
        div[data-testid="stVerticalBlock"] {{
            padding: 0; /* Adjust as necessary */
            margin: 0; /* Remove default margin */
        }}

        /* Targeting empty divs specifically */
        div[data-testid="stEmpty"] {{
            display: none; /* Hide empty divs */
        }}

        /* Adjust spacing for specific elements if needed */
        .stButton, .stTextInput, .stSelectbox {{
            margin-top: 0; /* Remove top margins for buttons and inputs */
            margin-bottom: 0; /* Remove bottom margins for buttons and inputs */
        }}

        /* Optional: Adjust spacing for headers and other components */
        h1, h2, h3, h4, h5, h6 {{
            margin: 0; /* Remove margins for headers */
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# Example usage
# add_bg_from_local("path/to/your/image.png")


def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Make the content more readable with a semi-transparent white background */
        .stMarkdown, .stHeader, div[data-testid="stVerticalBlock"] > div {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 0; /* Remove default margin */
        }}

        /* Remove padding from the main container */
        .stContainer {{
            padding: 0; /* Adjust as necessary */
        }}

        /* Optionally, remove padding from specific elements */
        div[data-testid="stVerticalBlock"] {{
            padding: 0; /* Adjust as necessary */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def page_one():
    # Adding some sample content for Page One

    st.title('Insurance prediction app')

    df = pd.read_csv('train.csv')

    Gender = st.selectbox("Gender", pd.unique(df["Gender"]))
    Age = st.selectbox("Age", pd.unique(df["Age"]))
    Driving_License = st.selectbox("Driving_License", pd.unique(df["Driving_License"]))
    Region_Code = st.selectbox("Region_Code", pd.unique(df["Region_Code"]))
    Previously_Insured = st.selectbox("Previously_Insured", pd.unique(df["Previously_Insured"]))
    Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df["Vehicle_Age"]))

    Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df["Vehicle_Damage"]))
    Annual_Premium = st.number_input('Annual_Premium')
    Policy_Sales_Channel = st.selectbox("Policy_Sales_Channel", pd.unique(df["Policy_Sales_Channel"]))
    Vintage = st.number_input('Vintage')

    inputs = {
        'Gender': Gender,
        'Age': Age,
        'Driving_License': Driving_License,
        'Region_Code': Region_Code,
        'Previously_Insured': Previously_Insured,
        'Vehicle_Age': Vehicle_Age,
        'Vehicle_Damage': Vehicle_Damage,
        'Annual_Premium': Annual_Premium,
        'Policy_Sales_Channel': Policy_Sales_Channel,
        'Vintage': Vintage
    }

    model = joblib.load('insurance_model_pipeline_hyper1.pkl')

    if st.button('Predict'):
        x_input = pd.DataFrame(inputs, index=[0])
        prediction = model.predict(x_input)
        st.write(' Predicted value is ::')
        st.write(prediction)


def page_two():
    st.subheader('Please upload a csv file for prediction :')
    upload_file = st.file_uploader('Choose a csv file ', type=['csv'])
    model = joblib.load('insurance_model_pipeline_hyper1.pkl')

    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.write('File uploaded successfully !!')
        st.write(df.head(2))
        if st.button('Predict for the uploaded file'):
            df['Response'] = model.predict(df)
            st.write(' Predicted value is ::')
            st.write(df['Response'])
            st.download_button(label='Download predicted results', data=df.to_csv(index=False),
                               mime='text/csv',
                               file_name='insurance_predict_output.csv')

def main():
    # Set page configuration
    st.set_page_config(
        page_title="My Streamlit App",
        layout="wide"
    )

    # Add background image - choose one of the following methods:

    # 1. From local file (uncomment and specify your image path)
    add_bg_from_local('insurance_predict_image.jpg')

    # 2. From URL (uncomment and specify your image URL)
    # add_bg_from_url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366")

    st.sidebar.title("Navigation")

    # Radio button for navigation
    page = st.sidebar.radio("Go to", ["Single Prediction", "Bulk Prediction"])

    # Navigation logic
    if page == "Single Prediction":
        page_one()
    else:
        page_two()


if __name__ == "__main__":
    main()