import streamlit as st
import requests
import pandas as pd
import json
import urllib


from pandas import json_normalize

# from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import requests
from io import BytesIO
import streamlit as st
import base64
import os

# import folium
# from streamlit_folium import folium_static
from datetime import datetime, timedelta
from PIL import Image as pilIm
from PIL import  ImageDraw
# import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
# import matplotlib.pyplot as plt
from io import BytesIO

def create_map_image(dataframe):
    gdf = gpd.GeoDataFrame(
        dataframe,
        geometry=gpd.points_from_xy(dataframe.longitude, dataframe.latitude),
        crs='EPSG:4326'
    )
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    gdf.plot(ax=ax, marker='o', color='red', markersize=100)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
    ax.set_axis_off()
    # Annotate each point with its index or other identifier
    # for idx, row in gdf.iterrows():
    #     ax.annotate(str(idx), xy=(row.geometry.x, row.geometry.y), xytext=(3,3), textcoords="offset points", color='blue')

    plt.tight_layout()

    # Save the map to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf


# Assuming df_comparables is a DataFrame with the Latitude and Longitude columns

# Then use the map_image_path as needed, for example in a PDF or Streamlit app.

def add_comparables_page(elements, image_buffer, dataframe,styles):
    # elements = []

    # Add the map image
    img = Image(image_buffer)
    img._restrictSize(6 * inch, 4 * inch)  # Adjust size as needed
    # elements.append(Spacer(1, 12))
    # Prepare the table data excluding latitude and longitude
    df_to_display = dataframe.drop(columns=['latitude', 'longitude'])
    table_data = [df_to_display.columns.tolist()] + df_to_display.values.tolist()


    # Prepare the table data
    # table_data = [dataframe.columns.tolist()] + dataframe.values.tolist()

    # Create and style the table
    table = Table(table_data)
    
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        # ('LEFTPADDING', (0, 0), (-1, -1), 3),
        # ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))
    elements.append(Paragraph("Comparable Property List", styles['Title']))
    elements.append(img)

    elements.append(table)

    # Append elements to the document
    # doc.build(elements)




# ################## crime ##########################
# Fetch crime data from API
def fetch_crime_data(zip_code):
    url = "https://crime-data-by-zipcode-api.p.rapidapi.com/crime_data"
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "crime-data-by-zipcode-api.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params={"zip": zip_code})
    response
    return response.json() if response.status_code == 200 else None

# Create doughnut chart
def create_doughnut_chart(data, title, filename):
    labels = list(data.keys())
    sizes = list(data.values())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=plt.cm.tab20.colors)
    # Draw a circle at the center of pie to make it look like a doughnut
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')  
    plt.tight_layout()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Create stacked bar chart
def create_stacked_bar_chart(data, title):
    fig, ax = plt.subplots()
    categories = list(data.keys())
    subcategories = list(data[categories[0]].keys())
    bar_width = 0.35
    indices = range(len(categories))
    bottoms = [0] * len(categories)

    for subcat in subcategories:
        values = [data[cat][subcat] for cat in categories]
        ax.bar(indices, values, bar_width, label=subcat, bottom=bottoms)
        bottoms = [bottoms[i] + values[i] for i in range(len(bottoms))]
    
    ax.set_xlabel('Categories')
    ax.set_ylabel('Rates')
    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()

    plt.tight_layout()
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the data

    plt.close(fig)
    return img_data
    # plt.savefig(filename)
    # plt.close()


##########################################################   end crime ##############
def generate_chart(df):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Days on Market'], df['Sold Price'], color='blue')
    ax.set_title('Price vs. Days on Market')
    ax.set_xlabel('Days on Market')
    ax.set_ylabel('Price')
    plt.tight_layout()

    # Save the figure to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the data

    plt.close(fig)  # Close the plot to free up memory
    return img_data


def generate_price_trend(df):
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['price'], marker='o', color='blue')
    ax.set_title('Price Trend Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    # plt.grid(True)
    plt.tight_layout()

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)
    return img_data

def generate_price_change_distribution(df):
    fig, ax = plt.subplots()
    df['priceChangeRate'].hist(bins=10, color='skyblue', ax=ax)
    ax.set_title('Distribution of Price Changes')
    ax.set_xlabel('Price Change Rate')
    ax.set_ylabel('Frequency')
    plt.tight_layout()

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)
    return img_data

def generate_price_vs_sqft(df):
    fig, ax = plt.subplots()
    ax.scatter(df['price'], df['pricePerSquareFoot'], alpha=0.5)
    ax.set_title('Price vs. Square Foot')
    ax.set_xlabel('Price')
    ax.set_ylabel('Price Per Square Foot')
    # plt.grid(True)
    plt.tight_layout()

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)
    return img_data

def generate_tax_chart(tax_df):
    fig, ax1 = plt.subplots()

    # Create bar for tax paid values
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('Tax Paid', color=color)
    ax1.bar(tax_df['time'], tax_df['taxPaid'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create line plot for property value
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Property Value', color=color)
    ax2.plot(tax_df['time'], tax_df['value'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    # Save the figure to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the data
    plt.close(fig)  # Close the plot to free up memory

    return img_data

# Function to generate chart from data
def generate_demographics_chart(variables, title):
    # Prepare data
    categories = [var['name']['en'] for var in variables.values()]
    values = [var['value'] for var in variables.values()]

    # Create doughnut chart
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a doughnut.
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig1.gca().add_artist(centre_circle)

    img_data1 = BytesIO()
    plt.savefig(img_data1, format='png', bbox_inches='tight')
    img_data1.seek(0)
    plt.close(fig1)

    # Create stacked bar chart
    fig2, ax2 = plt.subplots()
    ax2.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax2.set_title(title)
    ax2.set_ylabel('Values')

    img_data2 = BytesIO()
    plt.savefig(img_data2, format='png', bbox_inches='tight')
    img_data2.seek(0)
    plt.close(fig2)

    return img_data1, img_data2


def generate_event_frequencies(df):
    fig, ax = plt.subplots()
    df['event'].value_counts().plot(kind='bar', color='orange', ax=ax)
    ax.set_title('Frequency of Price Events')
    ax.set_xlabel('Event Type')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close(fig)
    return img_data

# Define function to get resized images

def generate_avg_price_by_location(df):
    avg_prices = df.groupby('Address')['Sold Price'].mean().sort_values()
    fig, ax = plt.subplots()
    avg_prices.plot(kind='bar', ax=ax, color='green')
    ax.set_ylabel('Average Price')
    plt.tight_layout()

    plt.title('Average Price by Location')
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the data

    plt.close(fig)  # Close the plot to free up memory
    return img_data
# Define function to get resized images
def get_resized_image(path, width=1*inch, height=1*inch):
    try:
        img = Image(path, width=width, height=height)
    except Exception as e:
        img = None
    return img


def create_adjustment_notes_table(notes):
    notes_list = [note.strip() for note in notes.split(',')]
    key_value_data = [note.split(':') for note in notes_list if ':' in note]

    # Header for two columns
    table_data = [['Adjustment Notes', '']]  # Initial header row with two cells
    for item in key_value_data:
        if len(item) == 2:
            key, value = item[0].strip(), item[1].strip()
        else:
            key, value = item[0].strip(), ''
        table_data.append([key, value])

    note_table = Table(table_data, colWidths=[1.5*inch, 1.5*inch])
    note_table.setStyle(TableStyle([
        ('SPAN', (0, 0), (-1, 0)),  # Span the header across all columns
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return note_table


def create_metric_cards(stats_data):
    """Function to create metric cards for statistics."""
    card_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.white),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BORDERRADIUS', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12)
    ])

    # Create a card for each statistic
    card_elements = []
    for stat, value in stats_data:
        data = [[stat, value]]
        card = Table(data, colWidths=[2*inch, 3*inch])
        card.setStyle(card_style)
        card_elements.append(card)
        card_elements.append(Spacer(1, 0.2 * inch))

    return card_elements


def convert_to_number(s):
    try:
        return int(s)
    except ValueError:
        return
    

def add_overall_crime_metrics(elements, overall_data, styles):
    # Define data for the table
    data = [
        ['Metric', 'Value'],
        ['Overall Crime Grade', overall_data['Overall Crime Grade']],
        ['Violent Crime Grade', overall_data['Violent Crime Grade']],
        ['Property Crime Grade', overall_data['Property Crime Grade']],
        ['Other Crime Grade', overall_data['Other Crime Grade']],
        ['Fact', overall_data['Fact']],
        ['Risk', overall_data['Risk']],
        ['Risk Detail', overall_data['Risk Detail']]
    ]

    # Enhance styles with bold text and highlights
    highlight_style = styles['Normal'].clone('highlight_style')
    highlight_style.textColor = colors.red
    highlight_style.backColor = colors.lightgrey
    highlight_style.fontSize = 11
    highlight_style.leading = 14
    highlight_style.bold = True

    # Modify data to include styles
    for i in range(1, len(data)):
        data[i][0] = Paragraph('<b>{}</b>'.format(data[i][0]), highlight_style)
        data[i][1] = Paragraph(data[i][1], styles['Normal'])

    # Create table with specified column widths
    table = Table(data, colWidths=[2.5*inch, 4.5*inch])

    # Styling the table with colors, bolds, and card-like sections
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),  # Header row background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header row text color
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Alignment of all cells
        ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font size for all cells
        # ('BOTTOMPADDING', (0, 0), (-1, -1), 12),  # Padding below cells
        # ('TOPPADDING', (0, 0), (-1, -1), 12),  # Padding above cells
        ('BOX', (0, 0), (-1, -1), 2, colors.darkgrey),  # Card-like box around the table
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertical alignment
        ('ROWBACKGROUNDS', (1, -1), [-1, -1], [colors.lightblue, colors.lightgreen])  # Alternating row backgrounds
    ]))

    # Add a heading and the table to the elements list
    elements.append(Paragraph('Overall Crime Metrics', styles['Title']))
    # elements.append(Spacer(1, 12))  # Space before the table
    elements.append(table)
    # elements.append(Spacer(1, 12))  # Space after the table


def generate_crime_charts(crime_data):
    # File paths for the charts
    charts = {}
    
    # Function to create and save a doughnut chart
    def create_doughnut_chart(data, title, filename):
        labels = list(data.keys())
        sizes = [float(data[key]) for key in labels]
        # Create a pie chart with a hole in the middle
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        for w in wedges:  # Make the wedges look nicer
            w.set_linewidth(2)
            w.set_edgecolor('white')
        for text in texts:  # Change the label styles
            text.set_color('grey')
            text.set_fontsize(8)
        for autotext in autotexts:  # Change the percentage styles
            autotext.set_color('white')
            autotext.set_fontsize(8)
        # Draw a circle at the center to turn the pie into a doughnut
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(title, pad=20)
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box
        plt.close()
        return filename
    
    # Function to create and save a horizontal bar chart
    def create_horizontal_bar_chart(data, title, filename):
        labels = list(data.keys())
        values = [float(data[key]) for key in labels]
        fig, ax = plt.subplots()
        bars = ax.barh(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Percent (%)')
        ax.set_title(title)
        
        for bar in bars:  # Add text labels to the end of each bar
            width = bar.get_width()
            label_x_pos = width + 1  # Adjust this value to move the label left or right
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, '%.1f%%' % width, va='center')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename
    
    # Assume 'crime_data' is a dictionary with the necessary structure.
    # Here's how you might use these functions to create your charts:
    # You'd call the functions with the appropriate data like this:
    charts['property_doughnut'] = create_doughnut_chart(crime_data['Crime BreakDown'][1]['Property Crime Rates'], 'Property Crime Doughnut', 'property_doughnut.png')
    charts['violent_doughnut'] = create_doughnut_chart(crime_data['Crime BreakDown'][0]['Violent Crime Rates'], 'Violent Crime Doughnut', 'violent_doughnut.png')
    charts['other_doughnut'] = create_horizontal_bar_chart(crime_data['Crime BreakDown'][2]['Other Crime Rates'], 'Other Level Bar', 'education_bar.png')
    
    return charts




def crop_to_circle(source_path, output_path, size=(100, 100)):
    img = pilIm.open(source_path).resize(size, pilIm.ANTIALIAS)
    mask = pilIm.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask) 
    draw.ellipse((0, 0) + img.size, fill=255)
    img.putalpha(mask)
    img.save(output_path)

# Note: You need to replace 'crime_data' with the actual data you have.
# This is just a template and will not run as-is because 'crime_data' is undefined here.
def footer(canvas, doc, agent_details, agent_image_stream, company_logo_stream):
    styles = getSampleStyleSheet()
    
    # Crop the agent's image to a circle and save temporarily
    agent_circular_img_path = 'bob.png'
    # crop_to_circle(agent_image_path, agent_circular_img_path, size=(50, 50))  # size is an example, adjust as needed

    if agent_image_stream:
        agent_image = Image(agent_image_stream)
    else:
        agent_image = Image('bob.png')  # Fallback to a default image

    if company_logo_stream:
        company_logo = Image(company_logo_stream)
    else:
        company_logo = Image('265900.png')  # Fallback to a default image

    # Modify the agent details paragraph with the updated information
    agent_info_paragraph = Paragraph(f'''
        <font size=10>
        <b>{agent_details['name']}</b><br/>
        {agent_details['company']}<br/>
        {agent_details['phone']}<br/>
        {agent_details['email']}
        </font>
    ''', styles["Normal"])
    # Agent details and image
    # agent_info_paragraph = Paragraph('''
    #     <font size=10>
    #     <b>BOB COOMARASWAMY</b><br/>
    #     ROYAL LEPAGE IGNITE REALTY, BROKERAGE<br/>
    #     416-282-3333<br/>
    #     bobbalakumar@royallepage.ca
    #     </font>
    # ''', styles["Normal"])
    
    # Load images
    # agent_image = Image(agent_circular_img_path)
    # company_logo = Image(company_logo_path)

    # Create the footer table
    footer_table = Table([
        [agent_image, agent_info_paragraph, company_logo]
    ], colWidths=[0.6*inch, None, 5*inch])
    
    # Style the footer table
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))

    
# Get the page size
    page_width, page_height = letter
    
    # Calculate the x coordinate for the footer
    x_coordinate = doc.leftMargin
    
    # Calculate the y coordinate for the footer; position it at the bottom
    y_coordinate = doc.bottomMargin
    
    # Draw the footer table on canvas
    footer_table.wrapOn(canvas, page_width, page_height)
    footer_table.drawOn(canvas, x_coordinate, y_coordinate)
    # Calculate width and height of the footer
    # footer_width, footer_height = footer_table.wrap(doc.width, doc.bottomMargin)
    # footer_table.drawOn(canvas, doc.leftMargin, footer_height)

# Function to create PDF
def create_pdf(df, dfs, taxDFs, crime_data, agent_image_stream, company_logo_stream,df_comparables,agent_details):
    # Initialize PDF
#  def create_pdf(df):
    # Initialize PDF
    pdf_path = "CMA_Report.pdf"
    pdf = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        # bottomMargin=1*inch,  # Increase the bottom margin to make space for the footer
    )
    elements = []
    styles = getSampleStyleSheet()

    # Define the headers for the new table layout
    # headers = ['Image', 'Address', 'Type', 'Bedrooms', 'Bathrooms', 'Living Area', 'Lot Size', 'Property Tax', 'Year Built','Days on Market',  'Sold Price', 'Adjusted Price']
    headers = [
        'Image', 'Address', 'Type', 'Bedrooms', 'Bathrooms', 'Living Area', 'Lot Size',
        'Property Tax', 'Year Built', 'Days on Market', 'Sold Price', 'Adjusted Price',
          'Garage', 'Drive', '#Park Spaces', 'Heat Type', 
        'Pool', 'Waterfront', 'Exterior'
    ]

    # Prepare data in a row-wise format with headers as the first column
    data = []
    for header_index, header in enumerate(headers):
        row = [Paragraph('<b>{}</b>'.format(header), styles['Normal'])]  # Header with bold style
        for index, item in df.iterrows():
            content = item[header]
            if header == 'Image' and content:
                row.append(get_resized_image(content))
            elif header == 'Address':
                # Wrap text in Address column
                row.append(Paragraph(content, styles['Normal']))
            elif header == 'Type':
                # Bold and color based on 'Subject' or 'Comparable'
                if content.lower() == 'subject':
                    row.append(Paragraph('<b><font color=red>{}</font></b>'.format(content), styles['Normal']))
                else:
                    row.append(Paragraph('<b><font color=blue>{}</font></b>'.format(content), styles['Normal']))
            
            else:
                row.append(content)
        data.append(row)

    # Create Table with the data
    # table = Table(data, colWidths=[None, 200, None, None, None, None, None, None, None])  # Set custom width for Address column
    table = Table(data, colWidths=[80, 80, 80, 80, 80, 80, 80, 80, 80])  # Set custom width for Address column

    # Styling the table
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
    ]))

    # Add the table to the elements list
    elements.append(Paragraph("Comparable Property Side by Side", styles['Title']))
    elements.append(table)
    elements.append(PageBreak())



    ##############crime #############

    add_overall_crime_metrics(elements, crime_data['Overall'], styles)

    chart_filenames = generate_crime_charts(crime_data)
    
    # Create image objects and add them to a table for side-by-side layout
    crime_images = [Image(chart_filenames[type]) for type in ['property_doughnut', 'violent_doughnut', 'other_doughnut']]
    # for image in crime_images:
    #     image._restrictSize(3*inch, 3*inch)  # Resize images if needed
    
    crime_images[0]._restrictSize(3*inch, 3*inch)  # Resize images if needed
    crime_images[1]._restrictSize(3*inch, 3*inch)  # Resize images if needed
    crime_images[2]._restrictSize(6*inch, 2*inch)  # Resize images if needed

    chart_table = Table([crime_images[:2]])  # Place three charts in one row
    chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    elements.append(chart_table)
    elements.append(crime_images[2])

    elements.append(PageBreak())

###################################

################  comparables table ###################

    image_buffer = create_map_image(df_comparables)

     # Add the comparables page to the PDF
    add_comparables_page(elements, image_buffer, df_comparables,styles)

    elements.append(PageBreak())

##################################

    # demographics = json.loads(demographics)
    # demographics
    # for category, details in demographics['data']['attributes'].items():
    #     try:
    #         if 'variables' in details and isinstance(details['variables'], dict):
    #             img_data1, img_data2 = generate_demographics_chart(details['variables'], details['category_name']['en'])
    #             elements.append(Paragraph(details['category_name']['en'], styles['Heading3']))

    #             # Create a table to display charts side by side
    #             chart1_image = Image(img_data1, 3*inch, 3*inch)
    #             chart2_image = Image(img_data2, 3*inch, 3*inch)
    #             chart_table = Table([[chart1_image, chart2_image]], colWidths=[3*inch, 3*inch])
    #             chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #                                             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    #             elements.append(chart_table)
    #             # elements.append(Spacer(1, 0.5*inch))
    #     except Exception as e:
    #         st.error(f"Failed to process category {category}: {str(e)}")
    


    comparable_df = df[df['Type'] == 'Comparable']
    low_price = comparable_df['Adjusted Price'].min()
    mean_price = comparable_df['Adjusted Price'].mean()
    high_price = comparable_df['Adjusted Price'].max()
    median_price = comparable_df['Adjusted Price'].median()

    stats_data = [
        ['Statistic', 'Value'],
        ['Low Price', f"${low_price:,}"],
        ['Mean Price', f"${mean_price:,.2f}"],
        ['High Price', f"${high_price:,}"],
        ['Median Price', f"${median_price:,}"]
    ]
     # Generate metric cards for the statistics
    metric_cards = create_metric_cards(stats_data)
    elements.append(Paragraph("Comparable Properties Statistics", styles['Title']))
    elements.extend(metric_cards)  # Add all metric cards to the document
    elements.append(PageBreak())


    # Create individual pages for each property
    for idx, row in df.iterrows():
        # Add property title
        elements.append(Paragraph(f"{row['Type']} - {row['Address']}", styles['Title']))
        elements.append(Spacer(1, 0.25 * inch))
        
        # Property details with headers
        details = [['Property Details', '']] + [[key, row[key]] for key in ['Bedrooms', 'Bathrooms', 'Living Area', 'Lot Size', 'Sold Price']]
        details_table = Table(details, colWidths=[1.5*inch, 1.5*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('SPAN', (0, 0), (-1, 0)),  # Span the header across all columns
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT')
            
        ]))

        # Adjustment notes table
        if pd.notnull(row['Adjustment Notes']):
            adjustment_notes_table = create_adjustment_notes_table(row['Adjustment Notes'])
        else:
            adjustment_notes_table = Table([['No Adjustment Notes', '']], colWidths=[1.5*inch, 1.5*inch])
            adjustment_notes_table.setStyle(TableStyle([
                ('SPAN', (0, 0), (-1, 0)),  # Span for 'No Adjustment Notes'
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT')
                
            ]))
        # Combine details and notes side by side
        combined_table = Table([[details_table, adjustment_notes_table]], colWidths=[3*inch, 3*inch])
        combined_table.setStyle(TableStyle([
            # ('SPAN', (0, 0), (-1, 0)),  # Span for 'Property Details'
            # ('SPAN', (1, 0), (-1, 0)),  # Span for 'Adjustment Notes'
            ('MARGIN', (0, 0), (-1, -1), 0.5*inch),
        ]))
        elements.append(combined_table)
        elements.append(Spacer(1, 0.25 * inch))
        # elements.append(PageBreak())
        images_data = []
        
        # Add property image if exists
        if pd.notnull(row['Image']):
            property_image = get_resized_image(row['Image'], width=3*inch, height=3*inch)
            images_data.append(property_image if property_image else 'No image available')

        # Add map image if exists
        if pd.notnull(row['staticMapImageUrl']):
            map_image = get_resized_image(row['staticMapImageUrl'], width=3*inch, height=3*inch)
            images_data.append(map_image if map_image else 'No map available')

        if images_data:
            # If we have one or two images, create a table to display them side by side
            images_table = Table([images_data], colWidths=[3*inch, 3*inch])
            elements.append(images_table)

        # Add description
        if pd.notnull(row['Description']):
            elements.append(Paragraph("Description:", styles['Heading2']))
            elements.append(Paragraph(row['Description'], styles['Normal']))

        elements.append(PageBreak())

        # elements.append(Paragraph(f"Pricing Statistics of {row['Address']}", styles['Title']))


        # img_data = generate_price_trend(dfs[idx])
        # chart_image_full = Image(img_data, 6*inch, 3*inch)  # Full-width chart
        # elements.append(chart_image_full)
        # elements.append(Spacer(1, 0.25*inch))

        # # Other charts in the second row
        # chart_functions = [generate_price_vs_sqft, generate_event_frequencies]
        # chart_images = [Image(func(dfs[idx]), 3*inch, 2*inch) for func in chart_functions]

        # # Row with three charts
        # row_charts_table = Table([chart_images], colWidths=[3*inch, 3*inch])
        # elements.append(row_charts_table)
        # # elements.append(PageBreak())
        # img_data = generate_price_change_distribution(dfs[idx])
        # chart_image_full = Image(img_data, 6*inch, 2*inch)  # Full-width chart
        # elements.append(chart_image_full)

        # elements.append(PageBreak())


        # tax_chart_img = generate_tax_chart(taxDFs[idx])
        # tax_chart = Image(tax_chart_img, 6*inch, 4*inch)
        # elements.append(Paragraph("Tax History", styles['Heading2']))
        # elements.append(tax_chart)
       



        
        # elements.append(PageBreak())


    # Build the PDF
    # pdf.build(elements,onFirstPage=footer, onLaterPages=footer)
    # pdf.build(elements, onFirstPage=lambda canvas, doc: footer(canvas, doc, agent_image_path, company_logo_path),
    #           onLaterPages=lambda canvas, doc: footer(canvas, doc, agent_image_path, company_logo_path))
    pdf.build(elements, onFirstPage=lambda canvas, doc: footer(canvas, doc, agent_details, agent_image_stream, company_logo_stream),
              onLaterPages=lambda canvas, doc: footer(canvas, doc, agent_details, agent_image_stream, company_logo_stream))
    # Clean up the temporary agent image
    # os.remove(agent_circular_img_path)
    return pdf_path

    
def get_image_stream(uploaded_file):
    if uploaded_file is not None:

        image = pilIm.open(uploaded_file)
        
        # Resize the image
        image = image.resize((50, 50), pilIm.Resampling.LANCZOS)
        # Construct the file path

        mask = pilIm.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse((0, 0) + image.size, fill=255)
        image.putalpha(mask)
        file_name =  uploaded_file.name
        file_path = os.path.join("./", file_name)
        
        # Save the resized image to the specified path
        image.save(file_path, format='PNG')  # Save as PNG format
        
        
        # Return the file path
        return file_path
    
    return None



def calculate_changes(df, base_index=0, columns_to_compare=[]):
    # Copy the DataFrame to avoid changing the original one
    df_with_changes = df.copy()
    
    # Get the base values from the specified base index
    base_values = df_with_changes.iloc[base_index]
    
    # Calculate the change for each specified column
    for col in columns_to_compare:
        # Define a new column name for changes
        change_col = f'{col} Change'
        
        # Subtract the base value from each value in the column
        df_with_changes[change_col] = df_with_changes[col] - base_values[col]
        
        # Format the change values with symbols
        df_with_changes[change_col] = df_with_changes[change_col]

    # Return the DataFrame with the changes
    return df_with_changes

# Helper function to format the change with arrows
def format_change(value):
    if value > 0:
        return f"+{value} \u2191"  # Up arrow for increase
    elif value < 0:
        return f"{value} \u2193"  # Down arrow for decrease
    return f"{value}"  # No change for zero


# Function to fetch property data from Zillow API
def fetch_similar_sold_properties(input_value):
    url = "https://zillow56.p.rapidapi.com/similar_sold_properties"
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],  # Using Streamlit secrets to store API key
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }

    querystring = input_value
    try:
        if "response" not in st.session_state:
            st.session_state.response = requests.get(url, headers=headers, params=querystring).json()
        # st.session_state.response.raise_for_status()
        return st.session_state.response
    except requests.RequestException as e:
        st.error(f"Failed to retrieve data: {str(e)}")
        return None
    
def fetch_property_details(zpid):
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




# Function to fetch demographics data
def fetch_demographics(lat, lng):
    API_HOST = "realty-in-ca1.p.rapidapi.com"
    
    API_ENDPOINT = "https://realty-in-ca1.p.rapidapi.com/properties/get-demographics"
    PARAMS = {"lng": lng, "lat": lat}
    PARAMS
    HEADERS = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": API_HOST
    }

    response = requests.get(API_ENDPOINT, headers=HEADERS, params=PARAMS)
    st.write(response)

    # response.json()
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to retrieve data.')
        return None

# def calculate_adjustment(comparable_property, user_inputs):
#     adjustments = sum(cost_matrix.get(improvement, 0) for improvement in user_inputs['improvements'])
#     adjustments -= sum(cost_matrix.get(deficiency, 0) for deficiency in user_inputs['deficiencies'])
#     adjustments += (comparable_property.get('livingAreaValue', 0) - user_inputs['subject_property']['Living Area']) * cost_matrix['sqFootage']
#     return adjustments


def calculate_adjustment(comparable_property, user_inputs,sublivingarea):
    adjustment_details = {}

    # Calculate improvements and deficiencies
    total_improvements = sum(cost_matrix.get(improvement, 0) for improvement in user_inputs['improvements'])
    total_deficiencies = sum(cost_matrix.get(deficiency, 0) for deficiency in user_inputs['deficiencies'])
    area_difference = comparable_property.get('livingAreaValue', 0) - sublivingarea
    area_adjustment = area_difference * cost_matrix['sq_ft_adjustment_per_unit']

    # Store individual components
    adjustment_details['total_improvements'] = total_improvements
    adjustment_details['total_deficiencies'] = total_deficiencies
    adjustment_details['area_difference'] = area_difference
    adjustment_details['area_adjustment'] = area_adjustment

    # Calculate total adjustments
    total_adjustment = total_improvements - total_deficiencies + area_adjustment
    adjustment_details['total_adjustment'] = total_adjustment

    return adjustment_details


def generate_search_query(row):
    # Define base URL and keywords
    base_url = "https://www.kijiji.ca/"
    keywords = ["price", "sqft"]
    # Construct the query part of the URL
    query_elements = [f"{row['Location']}", f"price per sqft {row['Square Footage']}"]
    query_elements.extend(keywords)  # Add static keywords
    query_string = " ".join(query_elements)
    
    # URL-encode the query string
    encoded_query = urllib.parse.quote(query_string)
    
    # Construct the full URL for the search
    full_url = f"{base_url}?q={encoded_query}"
    return full_url

# Generate search queries for each property

def fetchZPIDwithaddress( address):
    url = "https://zillow56.p.rapidapi.com/search_address"
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "zillow56.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params={"address": address})
    if response.status_code == 200:
        return response.json().get('zpid')
    else:
        st.error('Failed to retrieve subject property details.')
        return None


# Define your cost matrix for upgrades and downgrades
# 
st.title('Property Search and Comparison')
# client = OpenAI()
st.sidebar.title("Update Agent Information")
uploaded_agent_image = st.sidebar.file_uploader("Upload Agent Image", type=['png', 'jpg', 'jpeg'])
agent_details = {
    'name': st.sidebar.text_input("Agent's Name", value="BOB COOMARASWAMY"),
    'company': st.sidebar.text_input("Company Name", value="ROYAL LEPAGE IGNITE REALTY, BROKERAGE"),
    'phone': st.sidebar.text_input("Phone Number", value="416-282-3333"),
    'email': st.sidebar.text_input("Email", value="bobbalakumar@royallepage.ca"),
}
uploaded_company_logo = st.sidebar.file_uploader("Upload Company Logo", type=['png', 'jpg', 'jpeg'])


agent_image_stream = get_image_stream(uploaded_agent_image)
company_logo_stream = get_image_stream(uploaded_company_logo)



default_values = {
        'bedroom_value': 18000, 'kitchen_renovation': 25000, 'bathroom_value': 15000, 
        'foundation_repair': 30000, 'new_windows': 10000, 'heating_system': 8000,
        'cooling_system': 7000, 'landscaping': 5000, 'exterior_painting': 4000,
        'interior_painting': 3000, 'flooring_update': 6000, 'deck_installation': 10000,
        'pool_installation': 20000, 'needs_new_roof': -15000, 'old_windows': -10000,
        'foundation_issues': -30000, 'sq_ft_adjustment_per_unit': 200
    }
# Create three columns for the inputs
col1,  col3 = st.columns(2)

with col1:
    st.subheader("Listed Property Info")
    # zpid = st.text_input('Enter ZPID (Zillow Property ID)', value="28401079")
    # property_url = st.text_input('Enter Property URL', placeholder="Enter URL if you have")

    # with col2:
    # st.subheader("Known Inputs")
    address = st.text_input('Enter Address', value="8511 Seth Alexander Way, Bakersfield, CA 93306")
    # living_area = st.number_input('Enter living area of the subject property (sq ft)', value=1500)
    # improvements = st.multiselect('Select known improvements', list(cost_matrix.keys()))
    # deficiencies = st.multiselect('Select known deficiencies', list(cost_matrix.keys()))
        
    user_inputs = {
        'subject_property': {
            # 'Living Area': st.number_input('Enter living area of the subject property (sq ft)', value=1500)
        },
        'improvements': st.multiselect('Select known improvements', list(default_values.keys()), help="Select improvements that have been made to the property."),
        'deficiencies': st.multiselect('Select known deficiencies', list(default_values.keys()), help="Select any known deficiencies of the property.")
    }

    # input_value = {"zpid": zpid, "address": ""}


with col3:
    st.subheader("Cost Matrix Adjustments")
   
    cost_matrix = {}
    with st.expander("Edit Cost Matrix"):
        for key, value in default_values.items():
            cost_matrix[key] = st.number_input(f"{key.replace('_', ' ').title()} ($)", value=value, step=500)

# zpid = st.text_input('Enter ZPID', value="28401079")  # Default value for demonstration
# add = st.text_input('Enter Address', value="8511 Seth Alexander Way, Bakersfield, CA 93306")  # Default value for demonstration
# url = st.text_input('Enter URL',placeholder="optional")  # Default value for demonstration

# # Editable cost matrix


dfs = []
taxDFs = []

if st.button('Search'):
    subjectzpid = fetchZPIDwithaddress(address)
    similar_properties = fetch_similar_sold_properties({"zpid": subjectzpid})
    print(similar_properties)
    if similar_properties:
        detailed_data = []
        # Example data, replace with your actual data source
        comparable_properties_data = []

# Convert the list of dictionaries to a DataFrame

        subject_details = fetch_property_details(subjectzpid)  # Fetch details for the subject property
        # price_history = json_normalize(subject_details, record_path=['priceHistory'])
        pricehistorydf = json_normalize(subject_details, 'priceHistory', errors='ignore')
        taxhistorydf = json_normalize(subject_details, 'taxHistory', errors='ignore')

        dfs.append(pricehistorydf)
        taxDFs.append(taxhistorydf)
        # subject_details
        
        # Append subject property first if details are present
        if subject_details and 'resoFacts' in subject_details:
            sublat = subject_details.get('latitude', 'N/A')
            sublong = subject_details.get('longitude', 'N/A')


           
            # fetch demographical data ############
            # demographics = fetch_demographics(sublat, sublong)
            zip_code = subject_details.get('address', {}).get('zipcode', '77086')
            zip_code
            crime_data = fetch_crime_data(zip_code)
            

            reso_facts = subject_details['resoFacts']
            subject_data = {
                "Type":"Subject",
                "Image": subject_details.get('desktopWebHdpImageLink', 'N/A'),
                "Address": f"{subject_details.get('address', {}).get('streetAddress', 'N/A')}, {subject_details.get('address', {}).get('city', 'N/A')}, {subject_details.get('address', {}).get('state', 'N/A')} {subject_details.get('address', {}).get('zipcode', 'N/A')}",
                "Bedrooms": subject_details.get('bedrooms', 'N/A'),
                "Bathrooms": subject_details.get('bathrooms', 'N/A'),
                "Living Area": subject_details.get('livingArea', 'N/A'),
                "Lot Size": subject_details.get('lotSize', 'N/A'),
                "Year Built": subject_details.get('yearBuilt', 'N/A'),
                "Property Tax": subject_details.get('taxAssessedValue', 'N/A'),
                "Days on Market":  'N/A',
                "Sold Price":  'N/A',
                # "Price History": price_history(orient='records'),
                
                # "Adjusted Price": subject_details.get('zestimate', 0) + calculate_adjustment(subject_details, user_inputs),
                **{  # Additional details from resoFacts
                    "greenEnergyEfficient": reso_facts.get('greenEnergyEfficient', 'N/A'),
                    
                    "Fireplace": reso_facts.get('fireplaceFeatures', 'N/A'),
                    "Heat Type": ', '.join(reso_facts.get('heating', ['N/A'])),
                    "AIC": ', '.join(reso_facts.get('cooling', ['N/A'])),
                    "Garage": reso_facts.get('garage', 'N/A'),
                    "Drive": reso_facts.get('drive', 'N/A'),
                    "#Park Spaces": reso_facts.get('parkingSpaces', 'N/A'),
                    "Lot Size": reso_facts.get('lotSize', 'N/A'),
                    "Description": subject_details.get('description', 'N/A'),
                    "staticMapImageUrl": subject_details.get('staticMap', 'N/A')['sources'][0].get('url', 'N/A'),
                    # "Sq. Ft.": reso_facts.get('livingArea', 'N/A'),
                    "Exterior": ', '.join(reso_facts.get('exteriorFeatures', ['N/A'])),
                    "Pool": reso_facts.get('pool', 'N/A'),
                    "Waterfront": reso_facts.get('waterfront', 'N/A')

                }
            }
            detailed_data.append(subject_data)


        # Process similar properties
        for property in similar_properties.get("results", []):
            details = fetch_property_details(property['property']['zpid'])
            pricehistorydf = json_normalize(details, 'priceHistory', errors='ignore')
            proptaxhistorydf = json_normalize(details, 'taxHistory', errors='ignore')
            adjustment_metric = calculate_adjustment(details, user_inputs,  subject_details.get('livingArea', 0))
            dfs.append(pricehistorydf)
            taxDFs.append(proptaxhistorydf)
            # pricehistorydf
            if details and 'resoFacts' in details:
                reso_facts = details['resoFacts']
                flat_details = {
                    "greenEnergyEfficient": reso_facts.get('greenEnergyEfficient', 'N/A'),
                    "Fireplace": reso_facts.get('fireplaceFeatures', 'N/A'),
                    "Heat Type": ', '.join(reso_facts.get('heating', ['N/A'])),
                    "AIC": ', '.join(reso_facts.get('cooling', ['N/A'])),
                    "Garage": reso_facts.get('garage', 'N/A'),
                    "Drive": reso_facts.get('drive', 'N/A'),
                    "#Park Spaces": reso_facts.get('parkingSpaces', 'N/A'),
                    "Lot Size": reso_facts.get('lotSize', 'N/A'),
                    "Description": details.get('description', 'N/A'),
                    "staticMapImageUrl": details.get('staticMap', 'N/A')['sources'][0].get('url', 'N/A'),

                    # "Sq. Ft.": reso_facts.get('livingArea', 'N/A'),
                    "Exterior": ', '.join(reso_facts.get('exteriorFeatures', ['N/A'])),
                    "Pool": reso_facts.get('pool', 'N/A'),
                    "Waterfront": reso_facts.get('waterfront', 'N/A')
                }
                detailed_data.append({
                    "Type": "Comparable",
                    "Image": property['property'].get('compsCarouselPropertyPhotos', [{'mixedSources': {'jpeg': [{'url': 'N/A'}]}}])[0]['mixedSources']['jpeg'][0]['url'],
                    "Address": f"{property['property']['address']['streetAddress']}, {property['property']['address']['city']}, {property['property']['address']['state']} {property['property']['address']['zipcode']}",
                    "Bedrooms": property['property'].get('bedrooms', 'N/A'),
                    "Bathrooms": property['property'].get('bathrooms', 'N/A'),
                    "Living Area": property['property'].get('livingAreaValue', 'N/A'),
                    "Lot Size": property['property'].get('lotAreaValue', 'N/A'),
                    "Sold Price": property['property'].get('zestimate', 'N/A'),
                    "Days on Market": property['property'].get('daysOnZillow', 'N/A'),
                    "Year Built": property['property'].get('yearBuilt', 'N/A'),
                    "Property Tax": property['property'].get('taxAssessedValue', 'N/A'),
                    # "Price History": propprice_history(orient='records'),

                    # "Adjusted Price": subject_details.get('zestimate', 0) + calculate_adjustment(subject_details, user_inputs),
                    **flat_details
                })

                comparable_properties_data.append({
                    "Type": "Sold",
                    # "Image": property['property'].get('compsCarouselPropertyPhotos', [{'mixedSources': {'jpeg': [{'url': 'N/A'}]}}])[0]['mixedSources']['jpeg'][0]['url'],
                    "Address": f"{property['property']['address']['streetAddress']}, {property['property']['address']['city']}, {property['property']['address']['state']} {property['property']['address']['zipcode']}",
                    "Bedrooms": property['property'].get('bedrooms', 'N/A'),
                    "Bathrooms": property['property'].get('bathrooms', 'N/A'),
                    "Living Area": property['property'].get('livingAreaValue', 'N/A'),
                    "Lot Size": property['property'].get('lotAreaValue', 'N/A'),
                    "Sold Price": property['property'].get('zestimate', 'N/A'),
                    "Days on Market": property['property'].get('daysOnZillow', 'N/A'),
                    "homeType": property['property'].get('homeType', 'N/A'),
                    # "Property Tax": property['property'].get('taxAssessedValue', 'N/A'),
                    "latitude": property['property'].get('latitude', 'N/A'),
                    "longitude": property['property'].get('longitude', 'N/A')


                })

        # Create and display DataFrame
        df_comparables = pd.DataFrame(comparable_properties_data)
        # map_image_path = plot_properties(df_comparables)
        # st.image(map_image_path)
        # st.map(df_comparables,size=20,zoom=15)
        df = pd.DataFrame(detailed_data)
        columns_to_compare = ['Bedrooms', 'Bathrooms']
        df = calculate_changes(df, base_index=0, columns_to_compare=columns_to_compare)

        # dfs
        subject_property = df[df['Type'] == 'Subject'].iloc[0]
        df['Adjustment Notes'] = ""  # Initialize the column for notes

        # Process each row to calculate adjustments and generate notes
        for index, row in df.iterrows():
            if row['Type'] == 'Subject':
                # Skip adjustments for the subject property itself
                continue

            # Calculate the adjustment based on square footage differences
            square_footage_difference = subject_property['Living Area'] - row['Living Area']
            adjustment_amount = square_footage_difference * cost_matrix['sq_ft_adjustment_per_unit']



            bedroom_difference = subject_property['Bedrooms'] - row['Bedrooms']
            bedroom_adjustment = bedroom_difference * cost_matrix['bedroom_value']
            # detailed_adjustments.append(('Bedrooms', f"{bedroom_difference} Less rooms in basement" if bedroom_difference else "Same number of rooms", f"${bedroom_adjustment:,}"))

            # Bathroom count difference adjustment
            bathroom_difference = subject_property['Bathrooms'] - row['Bathrooms']
            bathroom_adjustment = bathroom_difference * cost_matrix['bathroom_value']
            # detailed_adjustments.append(('Washrooms', f"{bathroom_difference} Additional washrooms" if bathroom_difference else "Same number of washrooms", f"${bathroom_adjustment:,}"))

            adjusted_price = row['Sold Price'] + adjustment_amount
            df.at[index, 'Adjusted Price'] = adjusted_price
            # Generate the note for the current property
            note = (
                f"Original Price: ${row['Sold Price']:,} (Sold) ,\n\n"
                f"Approx Square Footage: {row['Living Area']} "
                f"{abs(square_footage_difference)} Sq Ft {'Lesser' if square_footage_difference < 0 else 'Greater'} ,\n\n"
                f"Adjustment Amount: ${abs(adjustment_amount):,} ,\n\n"
                f"Total Adjustment: ${adjustment_amount:,} ,\n\n"
                f"Adjusted Price: ${adjusted_price:,} ,\n\n"
            )

            for key, value in adjustment_metric.items():
                df.at[index, key] = value



            # st.write(note)
            # Append the note to the DataFrame
            df.at[index, 'Adjustment Notes'] = note


        comparable_df = df[df['Type'] == 'Comparable']

        low_price = comparable_df['Adjusted Price'].min()
        mean_price = comparable_df['Adjusted Price'].mean()
        high_price = comparable_df['Adjusted Price'].max()
        median_price = comparable_df['Adjusted Price'].median()

        # Display statistics
        # with st.expander("Comparable Properties Statistics"):
        #     st.write("Comparable Properties Statistics:")
        #     st.write(f"Low Price: ${low_price:,}")
        #     st.write(f"Mean Price: ${mean_price:,.2f}")
        #     st.write(f"High Price: ${high_price:,}")
        #     st.write(f"Median Price: ${median_price:,}")


        # st.dataframe(df, column_config={'Image': st.column_config.ImageColumn('Image'), 'Address': st.column_config.TextColumn('Address'), 'Adjustment Notes': st.column_config.TextColumn('Adjustment Notes'),"staticMapImageUrl": st.column_config.LinkColumn('staticMapImageUrl')})
        
        # sublat
        # sublong
        
        # crime_data
        
        # if st.button('Generate PDF'):
        pdf_pat = create_pdf(df, dfs,taxDFs, crime_data ,agent_image_stream, company_logo_stream, df_comparables, agent_details)
        
        # Display PDF in Streamlit
        with open(pdf_pat, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Provide a link to download the PDF
        st.download_button(
            label="Download PDF Report",
            data=base64_pdf,
            file_name="CMA_Report.pdf",
            mime='application/octet-stream'
        )

                
    else:
        st.error("Failed to retrieve properties.")
