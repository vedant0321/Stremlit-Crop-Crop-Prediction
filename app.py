import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from streamlit_navigation_bar import st_navbar
import pickle

st.set_page_config(initial_sidebar_state="collapsed")
# Load data
data = pd.read_csv("E://stramlit//Crop_recommendation.csv")



# Sidebar navigation

nav = ["Home", "Graphs", "Prediction", "Contact"]
styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}
nav = st_navbar(nav, styles=styles)
# nav=st_navbar( ["Home","Graphs", "Prediction", "Contact"])
# nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contact"])

# App title
st.title("Crop Prediction")

if nav == "Home":
    # Display image
    st.image("E://stramlit//Crop Prediction.jpg", use_column_width=True)
    
    # Checkbox to show/hide the data table
    if st.checkbox("Show tables"):
        st.write("Data Table Displayed Below (shown 1-100 attribute out of 2201): ")
        st.table(data.head(100))
    
   
if nav == "Graphs":
     # Graph type selection
    graph = st.selectbox("What kind of graph?", ["Interactive", "Non-Interactive"])
    if graph == "Interactive":
        graph_type = st.selectbox("Choose the type of graph", ["Pie", "Bar", "Scatter"])
        if graph_type == "Pie":
            attribute = st.selectbox("Select an attribute for the Pie Chart", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            fig = go.Figure(data=[go.Pie(labels=data["label"], values=data[attribute])])
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            st.plotly_chart(fig)
        elif graph_type == "Bar":
            fig = go.Figure()
            for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
                    fig.add_trace(go.Bar(x=data["label"], y=data[col], name=col))
                    fig.update_layout(barmode='group', xaxis_title="Crop", yaxis_title="Attribute")
                    st.plotly_chart(fig)
        
        elif graph_type == "Scatter":
            x_attribute = st.selectbox("Select the X-axis attribute", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], index=0)
            y_attribute = st.selectbox("Select the Y-axis attribute", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], index=1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data[x_attribute], y=data[y_attribute], mode='markers', text=data["label"]))
            fig.update_layout(xaxis_title=x_attribute, yaxis_title=y_attribute, title=f"{x_attribute} vs {y_attribute}")
            st.plotly_chart(fig)
                   
        # Using Plotly for interactive graphs
        
    else:
        # Using Matplotlib for non-interactive graphs
        layout = go.Layout(
            xaxis=dict(title="Crop"),
            yaxis=dict(title="Attribute"),
            title="Crop Attributes"
        ) 
        fig = go.Figure(layout=layout)
        for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            fig.add_trace(go.Bar(x=data["label"], y=data[col], name=col))
        st.plotly_chart(fig)

elif nav == "Prediction":
    # Prediction page content
    st.write("Welcome to the prediction page")
    # Step 1: Collect user input
   
    N = st.number_input("Enter Nitrogen value in range of 0-140:", min_value=0, max_value=140)
    P = st.number_input("Enter Phosphorus in range of 5-145:", min_value=5, max_value=145)
    K = st.number_input("Enter Potassium value in range of 5-205:", min_value=5,max_value=205)
    temperature= st.number_input("Enter Temperature value in range of 8.825675-43.67549:", min_value=8.825675,max_value=43.67549 ,format="%.2f")
    humidity = st.number_input("Enter Humidity value in range of 14.25804-99.98188:", min_value=14.25804,max_value=99.98188, format="%.2f")
    pH = st.number_input("Enter PH value in range of 3.504752-9.935091", min_value=3.504752, max_value=9.935091, format="%.2f")
    rainfall = st.number_input("Enter Rainfall (in mm) in range of 20.21127, max_value=298.5601:",min_value=20.21127, max_value=298.5601, format="%.2f")
    
    # Button to make prediction
    if st.button("Predict Crop"):
        # Step 2: Load the model (adjust the path as necessary)
        with open("E:\stramlit\model.pkl", "rb") as f:
            model = pickle.load(f) 
        # Step 3: Make prediction
        # Ensure the input data matches the model's expected format
        input_data = np.array([[N,P,K,temperature,humidity,pH,rainfall]])
        prediction = model.predict(input_data)
        # Step 4: Display prediction
        # Assuming prediction[0] is a numerical ID, and we have a dictionary that maps these IDs to crop names
        crop_names = {1: "Rice", 2: "maize", 3: "chikpea",4: "kidneybeans",5: "pigeonpeas",
        6: "mothbeans",7: "mungbeans",8: "blackgrams",9: " lentil",10: "pomogranate",11: "banana" 
        ,12 :"mango",13: "grapes ",14:"watermelon",15:"muskmeelon",16:"apple",17:"orange",18:"papaya",
        19:"coconut",20:"jute",21:"coffee"}  # Add all necessary mappings
        # Get the crop name using the prediction
        predicted_crop_name = crop_names.get(prediction[0], "Unknown crop")
        # Display the crop name
        st.write(f"The recommended crop is: {predicted_crop_name}")

elif nav == "Contact":
    # Contact page content
    st.write("Welcome to the contact page")