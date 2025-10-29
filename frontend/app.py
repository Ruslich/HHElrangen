import requests
import streamlit as st

st.title("Health Data Demo Frontend")

value = st.number_input("Enter a value to include in average calculation:", value=0)

if st.button("Submit"):
    response = requests.post("http://localhost:8000/avg", json={"value": value})
    if response.status_code == 200:
        st.success(f"Average calculated: {response.json().get('mean')}")
        st.write("Backend returned:", response.json())
    else:
        st.error("Error calculating average.")