import streamlit as st
import requests
import json
import os

# Function to fetch property details
def fetch_property_details(api_key, address):
    url = "https://zillow56.p.rapidapi.com/search_address"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params={"address": address})
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to retrieve subject property details.')
        return None

def search_property_details(zpid):
    url = "https://zillow56.p.rapidapi.com/property"
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params={"zpid": zpid})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to retrieve property details: {response.text}")
        return None



# Function to fetch comparables
def fetch_comparables(api_key, zpid):
    url = "https://zillow56.p.rapidapi.com/similar_sold_properties"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params={"zpid": zpid})
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to retrieve comparables.')
        return None

# Function to save data to a JSON file
def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    st.success(f'Data saved to {path}')

# Streamlit UI
st.title("Property Details Fetcher")

api_key = st.secrets["X-RapidAPI-Key"]
client_name = st.text_input("Client Name")
address_input = st.text_input("Enter the property address")

if st.button("Fetch Property Details"):
    if api_key and client_name and address_input:
        # Fetch subject property details
        subject_details = fetch_property_details(api_key, address_input)
        if subject_details:
            subject_zpid = subject_details.get('zpid')
            # Save subject property details
            subject_path = f'data/{client_name}/{subject_zpid}_subject.json'
            save_data(subject_details, subject_path)
            subject_zpid
            # Fetch and save comparables
            if subject_zpid:
                comps_data = fetch_comparables(api_key, subject_zpid)
                # comps_data
                if comps_data:
                    for comp in comps_data.get("results", []):
                        comp_details = search_property_details(comp['property']['zpid'])
                        comp_zpid = comp_details['zpid']
                        comp_path = f'data/{client_name}/{comp_zpid}_comparable.json'
                        save_data(comp_details, comp_path)
    else:
        st.error("Please fill in all fields.")
