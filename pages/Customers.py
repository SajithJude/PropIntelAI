import streamlit as st
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

def plot_map(dataframe):
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        dataframe,
        geometry=gpd.points_from_xy(dataframe.Longitude, dataframe.Latitude),
        crs='EPSG:4326'  # WGS 84 latitude and longitude
    )
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator for contextily

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, marker='o', color='red', markersize=50)  # Customize markers

    # Add contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
    ax.set_axis_off()
    plt.close(fig)
    
    return fig

st.title("Comparable Listings Map")

# Sample data input
data = {
    'Address': ['123 Elm St', '456 Maple St', '789 Oak St'],
    'Latitude': [37.77, 37.76, 37.75],
    'Longitude': [-122.42, -122.43, -122.44],
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
st.write("Data Preview:", df)

# Map plotting
if st.button('Plot Map'):
    fig = plot_map(df)
    st.pyplot(fig)
