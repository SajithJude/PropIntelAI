import streamlit as st

# Function to create a single metric display
def create_metric(label, value):
    st.metric(label=label, value=value)

# Layout of the dashboard
st.sidebar.title("REAL ESTATE")
st.sidebar.write("Chris Rivera\njoan.gilbert@gmail.com")
st.sidebar.button("Dashboard")
st.sidebar.button("Search")
st.sidebar.button("Customers")
st.sidebar.button("Payments")
st.sidebar.button("Settings")

st.title("Dashboard")

# Creating columns for the metrics
col1, col2, col3 = st.columns(3)

with col1:
    create_metric("Number of Customers", 238)

with col2:
    create_metric("Number of Generated PDFs", 156)

with col3:
    create_metric("Number of Searches", 258)
