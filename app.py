import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

st.set_page_config(page_title="Nigeria Environmental Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Final_Dataset_with_Government_Payments.csv")

df = load_data()
df['Mining_License_Total'] = df[['Valid', 'New', 'Renewed']].sum(axis=1)

# Sidebar filters
st.sidebar.title("Filters")
unique_states = df['subnational1'].unique().tolist()
selected_states = st.sidebar.multiselect(
    "Select up to 10 States", unique_states, default=unique_states[:5], max_selections=10
)

# Replace year selector with mineral selector
license_cols = ['Valid', 'New', 'Renewed']
mineral_cols = [col for col in df.columns if col.endswith("licenses") and col not in license_cols]
df_mineral = df[['subnational1'] + mineral_cols].copy()
df_melted = df_mineral.melt(id_vars='subnational1', var_name='Mineral', value_name='License Count')
df_melted['Mineral'] = df_melted['Mineral'].str.replace(" licenses", "", regex=False)
top_minerals = df_melted.groupby("Mineral")["License Count"].sum().sort_values(ascending=False).head(5).index.tolist()
available_minerals = sorted(df_melted['Mineral'].unique())
selected_minerals = st.sidebar.multiselect(
    "Select up to 10 Minerals", available_minerals, default=top_minerals, max_selections=10
)

# Apply filters
df = df[df['subnational1'].isin(selected_states)]

# Risk Score Calculation
conflict_cols = [col for col in df.columns if 'event_' in col]
fatality_cols = [col for col in df.columns if 'fatalities' in col.lower()]
govt_cols = [col for col in df.columns if col.startswith('govt_')]

df['Mining_Intensity'] = df[['Valid', 'New', 'Renewed']].sum(axis=1)
df['Total_Conflict_Events'] = df[conflict_cols].sum(axis=1)
df['Total_Conflict_Fatalities'] = df[fatality_cols].sum(axis=1)
df['Total_Govt_Payout'] = df[govt_cols].sum(axis=1)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(index=df.index)
df_scaled['Conflict_Norm'] = scaler.fit_transform(df[['Total_Conflict_Events']])
df_scaled['Fatalities_Norm'] = scaler.fit_transform(df[['Total_Conflict_Fatalities']])
df_scaled['Inverse_Payout_Norm'] = 1 - scaler.fit_transform(df[['Total_Govt_Payout']])
df_scaled['Mining_Intensity_Norm'] = scaler.fit_transform(df[['Mining_Intensity']])
df_scaled['Carbon_Emissions_Norm'] = scaler.fit_transform(df[['All Carbon Emissions']])
df_scaled['Tree_Emissions_Norm'] = scaler.fit_transform(df[['All Tree Emissions']])

weights = {
    'Conflict_Norm': 0.25,
    'Fatalities_Norm': 0.20,
    'Inverse_Payout_Norm': 0.15,
    'Mining_Intensity_Norm': 0.15,
    'Carbon_Emissions_Norm': 0.15,
    'Tree_Emissions_Norm': 0.10
}

df['Integrated_Risk_Score'] = sum(df_scaled[col] * w for col, w in weights.items())
df['Integrated_Risk_Zone'] = pd.qcut(df['Integrated_Risk_Score'], q=3, labels=['Low', 'Medium', 'High'])

# KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

card_style = """
<div style='background-color:#3b5f78;
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            height:100px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;'>
    <div style='font-size:26px; font-weight:bold; line-height:1;'>{value}</div>
    <div style='font-size:13px; margin-top:8px; line-height:1.2; word-break:break-word;'>{label}</div>
</div>
"""


with col1:
    st.markdown(card_style.format(value=f"{df['Mining_License_Total'].sum():,.0f}", label="Total Licenses"), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(value=f"{df['Total_Govt_Payout'].sum():,.0f}", label="Total Govt Payout"), unsafe_allow_html=True)
    

with col3:
    st.markdown(card_style.format(value=f"{df['Total_Conflict_Events'].sum():,.0f}", label="Total Conflict Events"), unsafe_allow_html=True)

with col4:
    st.markdown(card_style.format(value=f"{df['All Tree Emissions'].sum():,.0f}", label="Total Tree Emissions"), unsafe_allow_html=True)

with col5:
    st.markdown(card_style.format(value=f"{df['All Carbon Emissions'].sum():,.0f}", label="Total Carbon Emissions"), unsafe_allow_html=True)


st.markdown("---")

# Map and Bar Chart
col_map, col_chart = st.columns([1, 1])

state_coords = {
    'Abia': [5.4527, 7.5248], 'Adamawa': [9.3265, 12.3984], 'Akwa Ibom': [5.0359, 7.9120],
    'Anambra': [6.2209, 7.0723], 'Bauchi': [10.3116, 9.8442], 'Bayelsa': [4.7500, 6.0833],
    'Benue': [7.3369, 8.7400], 'Borno': [11.8333, 13.1500], 'Cross River': [5.8702, 8.5988],
    'Delta': [5.7041, 5.9335], 'Ebonyi': [6.3249, 8.1137], 'Edo': [6.6342, 5.9304],
    'Ekiti': [7.7184, 5.3101], 'Enugu': [6.5244, 7.5186], 'Gombe': [10.2897, 11.1673],
    'Imo': [5.5720, 7.0588], 'Jigawa': [12.2280, 9.5616], 'Kaduna': [10.5236, 7.4381],
    'Kano': [12.0022, 8.5919], 'Katsina': [12.9886, 7.6009], 'Kebbi': [11.4942, 4.2333],
    'Kogi': [7.7339, 6.6906], 'Kwara': [8.5000, 4.5500], 'Lagos': [6.5244, 3.3792],
    'Nasarawa': [8.4904, 8.5204], 'Niger': [9.9306, 5.5983], 'Ogun': [6.9980, 3.4737],
    'Ondo': [7.1000, 5.1500], 'Osun': [7.5629, 4.5199], 'Oyo': [7.8408, 3.9319],
    'Plateau': [9.2182, 9.5176], 'Rivers': [4.8431, 6.9114], 'Sokoto': [13.0059, 5.2476],
    'Taraba': [7.8704, 9.7810], 'Yobe': [12.2939, 11.4397], 'Zamfara': [12.1221, 6.2236],
    'FCT': [9.0579, 7.4951]
}

with col_map:
    m = folium.Map(location=[9.0820, 8.6753], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    top_states = df.sort_values(by='Mining_License_Total', ascending=False)['subnational1'].head(5).tolist()

    for _, row in df.iterrows():
        coords = state_coords.get(row['subnational1'])
        if not coords:
            continue
        color = 'red' if row['Integrated_Risk_Zone'] == 'High' else 'orange' if row['Integrated_Risk_Zone'] == 'Medium' else 'green'
        popup_html = f"""
            <b style='color:{color};'>{row['subnational1']}</b><br>
            Risk Zone: <b>{row['Integrated_Risk_Zone']}</b><br>
            Score: {row['Integrated_Risk_Score']:.2f}<br>
            Licenses: {int(row['Mining_License_Total'])}
        """
        folium.CircleMarker(
            location=coords,
            radius=5 + 10 * row['Integrated_Risk_Score'],
            color='black' if row['subnational1'] in top_states else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            tooltip=row['subnational1'],
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(marker_cluster)
    st_folium(m, use_container_width=True, height=500)

with col_chart:
    license_cols = ['Valid', 'New', 'Renewed']
    df_bar = df[['subnational1'] + license_cols + ['Mining_License_Total']]
    df_bar_top5 = df_bar.sort_values(by='Mining_License_Total', ascending=False).head(5)
    df_bar_melted = df_bar_top5.drop(columns='Mining_License_Total').melt(
        id_vars='subnational1', var_name='License Type', value_name='Count')
    
    fig1 = px.bar(df_bar_melted, x='subnational1', y='Count', color='License Type', barmode='stack')
    st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# Tabs
tab_minerals, tab_conflict, tab_env = st.tabs(["Minerals by State", "Conflicts Over Time", "Environment Over Time"])

with tab_minerals:

    # Filter and process data for selected minerals only
    df_mineral_filtered = df[['subnational1'] + mineral_cols].copy()
    df_melted_filtered = df_mineral_filtered.melt(id_vars='subnational1', var_name='Mineral', value_name='License Count')
    df_melted_filtered['Mineral'] = df_melted_filtered['Mineral'].str.replace(" licenses", "", regex=False)

    # Keep only selected minerals
    df_selected = df_melted_filtered[df_melted_filtered['Mineral'].isin(selected_minerals)]

    # Group and plot
    df_grouped = df_selected.groupby(["subnational1", "Mineral"])["License Count"].sum().reset_index()

    fig = px.bar(
        df_grouped,
        x="License Count",
        y="subnational1",
        color="Mineral",
        orientation="h",
        title="⛏️ Mineral Licenses by Selected States and Minerals",
        barmode="stack"
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_conflict:

    # Filter conflict data based on selected states
    df_conflict = df[df['subnational1'].isin(selected_states)][['subnational1'] + conflict_cols].copy()
    df_melted = df_conflict.melt(id_vars='subnational1', var_name='Event Type', value_name='Count')
    df_melted['Event Type'] = df_melted['Event Type'].str.replace("event_", "", regex=False)

    col1, col2 = st.columns(2)

    with col1:
        conflict_by_type = df_melted.groupby("Event Type")["Count"].sum().reset_index()
        fig_pie = px.pie(
            conflict_by_type,
            names="Event Type",
            values="Count",
            hole=0.4,
            title="Conflict Events by Type (Filtered by Selected States)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        df['Total_Conflict_Events'] = df[conflict_cols].sum(axis=1)
        top_conflict_states = (
            df[df['subnational1'].isin(selected_states)][['subnational1', 'Total_Conflict_Events']]
            .sort_values(by='Total_Conflict_Events', ascending=False)
            .head(10)
        )
        fig_bar = px.bar(
            top_conflict_states,
            x='subnational1',
            y='Total_Conflict_Events',
            color='Total_Conflict_Events',
            title="Top States by Conflict Events"
        )
        st.plotly_chart(fig_bar, use_container_width=True)


with tab_env:

    # Load external environment data
    env_external = pd.read_excel("merged_carbon_tree_data.xlsx")

    # Filter based on selected states
    env_filtered = env_external[env_external['subnational1'].isin(selected_states)]

    # Select carbon emission columns
    carbon_cols = [col for col in env_filtered.columns if "gfw_forest_carbon_gross_emissions_" in col and "__Mg_CO2e" in col]

    # Melt into long format
    carbon_time_df = env_filtered[["subnational1"] + carbon_cols]
    carbon_melted = carbon_time_df.melt(id_vars="subnational1", var_name="Year", value_name="Emissions")
    carbon_melted["Year"] = carbon_melted["Year"].str.extract(r'(\d{4})')
    carbon_melted = carbon_melted.dropna(subset=["Year"])
    carbon_melted["Year"] = carbon_melted["Year"].astype(int)

    # Aggregate emissions by year
    carbon_yearly = carbon_melted.groupby("Year")["Emissions"].sum().reset_index()

    # Plot
    fig_carbon = px.line(
        carbon_yearly,
        x="Year",
        y="Emissions",
        markers=True,
        title="Forest Carbon Emissions Over Time (Filtered by Selected States)",
        labels={"Emissions": "Emissions (Mg CO₂e)"}
    )
    st.plotly_chart(fig_carbon, use_container_width=True)


# Clean padding
st.markdown("<style>div.block-container { padding-bottom: 1rem; }</style>", unsafe_allow_html=True)