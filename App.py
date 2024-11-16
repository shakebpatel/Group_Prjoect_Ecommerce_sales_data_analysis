
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from geopy.geocoders import Nominatim

# Import the load_data function from your data.py


def load_data():
    df = pd.read_csv("NewUpdated.csv")

    # Convert date columns to datetime
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"])

    # Create time-based features
    df["Year"] = df["order_purchase_timestamp"].dt.year
    df["Month"] = df["order_purchase_timestamp"].dt.month

    # Mapping months to financial months
    month_dict = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
                  10: 7, 11: 8, 12: 9, 1: 10, 2: 11, 3: 12}
    df["Financial_Month"] = df["Month"].map(month_dict)

    # Create Financial_Year column and ensure the format is "Year - Year+1"
    df["Financial_Year"] = df.apply(
        lambda x: f"{x['Year']} - {x['Year'] +
                                   1}" if x['Month'] >= 4 else f"{x['Year']-1} - {x['Year']}",
        axis=1
    )

    # Ensure Financial_Year is treated as a string, which will prevent any floating-point format
    df["Financial_Year"] = df["Financial_Year"].astype(str)

    return df


# Set page layout
st.set_page_config(layout="wide")

# CSS for styling the background and container
background_image_url = "url(https://wallpapercave.com/wp/wp7566374.jpg)"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: {background_image_url};
        background-size: cover;
        background-position: center 20%;  /* Adjust the position of the background */
        background-repeat: no-repeat;
        height: 100vh;
        color: white;
        padding: 20px;
    }}
    .block-container {{
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
df = load_data()

# Create financial features
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["Year"] = df["order_purchase_timestamp"].dt.year
df["Month"] = df["order_purchase_timestamp"].dt.month
df["Financial_Month"] = df["Month"].map(
    {4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 1: 10, 2: 11, 3: 12}
)

with st.container():
    st.markdown(
        """
        <style>
        .top-filters {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }
        .top-filters div {
            flex: 1;
            margin: 0 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )
logo_path = os.path.join(
    'C:/Users/a.n.shaikh0129/OneDrive/Desktop/Masai/Project+/StreamLit', 'tech-2.jpeg'
)


with st.sidebar:
    if os.path.exists(logo_path):
        # Use the updated parameter
        st.image(logo_path, use_container_width=True)
    else:
        st.write("Logo file not found.")
# Create sidebar filters
with st.container():
    st.markdown("<div style='margin-top: 20px;'></div>",
                unsafe_allow_html=True)  # Adds space above the filters

    with st.expander("🔍 Show Filters", expanded=True):

        # "Select All" functionality for each filter
        select_all_years = st.checkbox("Select All Years", value=True)
        if select_all_years:
            selected_year = df["Year"].unique().tolist()
        else:
            selected_year = st.multiselect("Select Year", df["Year"].unique())

        select_all_states = st.checkbox("Select All States", value=True)
        if select_all_states:
            selected_retailer = df["customer_state"].unique().tolist()
        else:
            selected_retailer = st.multiselect(
                "Select State", df["customer_state"].unique())

        select_all_prices = st.checkbox("Select All Category", value=True)
        if select_all_prices:
            selected_company = df["product_category_name"].unique().tolist()
        else:
            selected_company = st.multiselect(
                "Select Category", df["product_category_name"].unique())


# Main content section
st.markdown('<div class="content" style="margin-top: 40px;">',
            unsafe_allow_html=True)
# Apply filters and update data
filtered_df = df[(df["Year"].isin(selected_year)) &
                 (df["customer_state"].isin(selected_retailer)) &
                 (df["product_category_name"].isin(selected_company))]


# Filter the data
filtered_df = df[(df["Year"].isin(selected_year)) &
                 (df["customer_state"].isin(selected_retailer)) &
                 (df["product_category_name"].isin(selected_company))]

# Main content section
st.title("E-Commerce Insights Dashboard")


def geocode_location(location):
    geolocator = Nominatim(user_agent="location_mapper")
    location_obj = geolocator.geocode(location)
    if location_obj:
        return location_obj.latitude, location_obj.longitude
    return None, None


# Function to get latitude and longitude for states
location_data = []
for state in filtered_df['customer_state'].unique():
    lat, lon = geocode_location(state)
    if lat and lon:
        location_data.append(
            {'state': state, 'latitude': lat, 'longitude': lon})

# Convert location data into a DataFrame
location_df = pd.DataFrame(location_data)

# Merge the latitude and longitude with the original filtered DataFrame
filtered_df = filtered_df.merge(
    location_df, left_on='customer_state', right_on='state', how='left')

# KPI Columns
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"${int(filtered_df['price'].sum()):,}")
col2.metric("Total Orders", f"{int(filtered_df['order_id'].nunique()):,}")
col3.metric("Total Customers", f"{
            int(filtered_df['customer_unique_id'].nunique()):,}")

# Calculate delivery time for each order
filtered_df['delivery_time'] = (pd.to_datetime(filtered_df['order_delivered_customer_date']) -
                                pd.to_datetime(filtered_df['order_approved_at'])).dt.days

# Calculate the average delivery time
average_delivery_time = filtered_df['delivery_time'].mean()

# Display the average delivery time in col4
col4.metric("Average Delivery Time", f"{int(average_delivery_time):,} days")


# Financial Month Sales Bar Chart
monthly_sales_by_financial = filtered_df.groupby(["Year", "Financial_Month"])[
    "price"].sum().unstack()

# Reset index and melt the dataframe for plotting
monthly_sales_df = monthly_sales_by_financial.reset_index().melt(
    id_vars="Year", var_name="Financial_Month", value_name="Sales"
)

# Create the bar chart with an attractive color palette
fig_bar = px.bar(
    monthly_sales_df,
    x="Financial_Month",
    y="Sales",
    color="Year",
    labels={"Financial_Month": "Financial Month", "Sales": "Total Sales ($)"},
    title="Monthly Sales by Financial Month",
    barmode="group",
    color_continuous_scale="Sunset",  # Beautiful color scale
)

# Customize the layout for better appearance
fig_bar.update_layout(
    xaxis_title="Financial Month",
    yaxis_title="Total Sales ($)",
    template="plotly_dark",  # Dark theme for a modern look
    title_x=0.5,  # Center the title
    title_font=dict(size=20),  # Increase the title font size

    yaxis=dict(tickprefix="$"),  # Add dollar sign prefix
    showlegend=True,  # Show legend for better understanding of the chart
    plot_bgcolor="rgb(32, 32, 32)",  # Dark background for the plot area
)

# Display the chart in Streamlit
st.plotly_chart(fig_bar, use_container_width=True)

# Retailer Revenue Analysis
col5, col6 = st.columns([1, 1])  # Keep equal column width

# Pie chart on the left column
col5, col6 = st.columns([1, 2])  # Make col6 larger by giving it a weight of 2

# Pie chart on the left column
col5, col6 = st.columns([1, 2])  # Make col6 larger by giving it a weight of 2

# Pie chart on the left column
col5, col6 = st.columns([1, 2])  # Make col6 larger by giving it a weight of 2

# Pie chart on the left column
with col5:
    st.subheader("Payment Method Distribution")
    payment_counts = filtered_df['payment_type'].value_counts()

    # Create a Plotly pie chart
    fig = px.pie(
        names=payment_counts.index,
        values=payment_counts.values,
        title="Payment Method Distribution",
        hole=0.3,  # Make it a donut chart if desired
    )
    # Customize the figure size, title, and layout
    fig.update_layout(
        autosize=False,  # Fixed size for consistency
        width=600,  # Increased width
        height=600,  # Increased height
        # Adjust margins for better display
        margin=dict(t=50, b=30, l=30, r=30),
        title={
            'text': "Payment Method Distribution",
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'y': 0.98,  # Move title higher to avoid overlap with the chart
            'yanchor': 'top',
        }
    )
    st.plotly_chart(fig)

# Bar chart on the right column
with col6:
    st.subheader("Product Categories By Order")
    Top_Categories = filtered_df['product_category_name'].value_counts().head(
        10)

    # Create a Plotly bar chart
    fig_company = px.bar(
        Top_Categories,
        title="Top 10 Product Categories By Orders",
        color=Top_Categories,
        color_continuous_scale="Viridis",
        orientation="h",
        text=Top_Categories
    )
    fig_company.update_layout(
        width=800,  # Increased width for a larger chart
        height=600,  # Height remains the same as the pie chart
        xaxis_title="Number of Orders",
        yaxis_title="Product Category",
        template="plotly_white",
        showlegend=False,
        # Increased top margin to avoid overlap
        margin=dict(t=50, b=30, l=30, r=30),
        title={
            'text': "Top 10 Product Categories By Orders",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,  # Move title higher to avoid overlap
            'yanchor': 'top'
        }
    )
    st.plotly_chart(fig_company)

# Averga ereview Score

# Calculate the average review score by product category
avg_review_scores = filtered_df.groupby('product_category_name')[
    'review_score'].mean().sort_values(ascending=False).head(10)

# Create a histogram using Plotly Express
fig = px.bar(  # Use a bar chart instead of a histogram for better control over color
    avg_review_scores,
    x=avg_review_scores.index,
    y=avg_review_scores.values,
    labels={'x': 'Product Category', 'y': 'Average Review Score'},
    title="Average Review Score Distribution by Product Category",
    color=avg_review_scores.index,  # Color bars based on product categories
    # Set color scale (you can choose any scale)
    color_continuous_scale="Viridis",
)

# Customize the figure layout
fig.update_layout(
    xaxis_title="Product Category",
    yaxis_title="Average Review Score",
    template="plotly_white",
    xaxis_tickangle=-45,  # Rotates x-axis labels for better readability
)

# Show the histogram in Streamlit
st.subheader("Average Review Score Distribution")
st.plotly_chart(fig)


# Map Plot for Customer Locations
st.subheader("Customer Location Map")
if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
    st.map(filtered_df)
else:
    st.write("Latitude and Longitude data not available for mapping.")
