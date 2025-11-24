
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Receivables Dashboard - Bulb Manufacturing",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    # Clean Amount column (handle commas)
    if 'Amount' in df.columns and df['Amount'].dtype == 'object':
        df['Amount'] = df['Amount'].astype(str).str.replace(',', '').astype(float)
    # Ensure proper dtypes
    if 'Inv_Date' in df.columns:
        df['Inv_Date'] = pd.to_datetime(df['Inv_Date'])
    if 'Due_Date' in df.columns:
        df['Due_Date'] = pd.to_datetime(df['Due_Date'])
    # Normalise text
    for col in ['whether_invoice_paid', 'RSM', 'ASM', 'Sales_Person', 'City', 'State', 'Name_of_the_dealer']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def add_ageing_buckets(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp(datetime.today().date())
    df = df.copy()
    df['Is_Unpaid'] = df['whether_invoice_paid'].str.upper().eq('NO')

    # Days to due and overdue days
    df['Days_To_Due'] = (df['Due_Date'] - today).dt.days
    df['Days_Overdue'] = (today - df['Due_Date']).dt.days

    def ageing_bucket(row):
        if not row['Is_Unpaid'] or pd.isna(row['Due_Date']):
            return 'Paid / NA'
        dtd = row['Days_To_Due']
        dov = row['Days_Overdue']
        if dtd >= 0 and dtd <= 7:
            return 'Due in 0–7 days'
        if dov >= 0 and dov <= 7:
            return 'Overdue 0–7 days'
        if dov >= 8 and dov <= 14:
            return 'Overdue 8–14 days'
        if dov > 14:
            return 'Overdue >14 days'
        if dtd > 7:
            return 'Due in >7 days'
        return 'Other'

    df['Ageing_Bucket'] = df.apply(ageing_bucket, axis=1)
    # Add Month_Year for trend analysis
    if 'Inv_Date' in df.columns:
        df['Month_Year'] = df['Inv_Date'].dt.to_period('M').astype(str)
    return df

def get_india_coordinates():
    """Returns dictionary of Indian states with their lat/lon coordinates"""
    return {
        'Andhra Pradesh': (15.91, 79.74), 'Arunachal Pradesh': (28.21, 94.72),
        'Assam': (26.20, 92.93), 'Bihar': (25.09, 85.31),
        'Chhattisgarh': (21.27, 81.86), 'Goa': (15.29, 74.12),
        'Gujarat': (22.25, 71.19), 'Haryana': (29.05, 76.08),
        'Himachal Pradesh': (31.10, 77.17), 'Jharkhand': (23.61, 85.27),
        'Karnataka': (15.31, 75.71), 'Kerala': (10.85, 76.27),
        'Madhya Pradesh': (22.97, 78.65), 'Maharashtra': (19.75, 75.71),
        'Manipur': (24.66, 93.90), 'Meghalaya': (25.46, 91.36),
        'Mizoram': (23.16, 92.93), 'Nagaland': (26.15, 94.56),
        'Odisha': (20.95, 85.09), 'Punjab': (31.14, 75.34),
        'Rajasthan': (27.02, 74.21), 'Sikkim': (27.53, 88.51),
        'Tamil Nadu': (11.12, 78.65), 'Telangana': (18.11, 79.01),
        'Tripura': (23.94, 91.98), 'Uttar Pradesh': (26.84, 80.94),
        'Uttarakhand': (30.06, 79.01), 'West Bengal': (22.98, 87.85),
        'Delhi': (28.70, 77.10), 'Chandigarh': (30.73, 76.77),
        'Jammu and Kashmir': (33.77, 76.57), 'Ladakh': (34.15, 77.57)
    }

def create_india_map(df: pd.DataFrame):
    """Creates an interactive India map showing state-level risk analysis"""
    if df.empty or 'State' not in df.columns:
        return None
    
    # Prepare state-level data
    state_data = df.groupby('State').agg(
        Total_Rev=('Amount', 'sum'),
        Unpaid_Amt=('Outstanding_Amount', 'sum')
    ).reset_index()
    
    # Calculate unpaid percentage
    state_data['Unpaid_Pct'] = np.where(
        state_data['Total_Rev'] > 0,
        (state_data['Unpaid_Amt'] / state_data['Total_Rev']) * 100,
        0
    )
    
    # Risk classification based on median threshold
    threshold = state_data['Unpaid_Pct'].median()
    state_data['Risk_Status'] = state_data['Unpaid_Pct'].apply(
        lambda x: 'High Risk (Bad Debt)' if x > threshold else 'Low Risk (Safe)'
    )
    
    # Map coordinates - handle state name variations
    india_coords = get_india_coordinates()
    
    def get_coords(state_name):
        """Try exact match first, then case-insensitive match"""
        if state_name in india_coords:
            return india_coords[state_name]
        # Try case-insensitive match
        state_lower = str(state_name).strip()
        for key, coords in india_coords.items():
            if key.lower() == state_lower.lower():
                return coords
        return (None, None)
    
    state_data['Lat'] = state_data['State'].apply(lambda x: get_coords(x)[0])
    state_data['Lon'] = state_data['State'].apply(lambda x: get_coords(x)[1])
    
    # Remove states without coordinates
    state_data = state_data.dropna(subset=['Lat', 'Lon'])
    
    if state_data.empty:
        return None
    
    # Create map
    fig = px.scatter_mapbox(
        state_data,
        lat="Lat",
        lon="Lon",
        color="Risk_Status",
        size="Total_Rev",
        color_discrete_map={'High Risk (Bad Debt)': 'red', 'Low Risk (Safe)': 'green'},
        hover_name="State",
        hover_data={
            'Lat': False,
            'Lon': False,
            'Total_Rev': ':,.0f',
            'Unpaid_Amt': ':,.0f',
            'Unpaid_Pct': ':.1f',
            'Risk_Status': True
        },
        zoom=4,
        center={"lat": 22.0, "lon": 78.0},  # Center of India
        height=600,
        title="India Sales Performance & Risk Map"
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    
    return fig

# ---------- SIDEBAR FILE PICKER ----------
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown(
    "Upload or place the **Excel file** (`bulb_sales_sample_1000_rows.xlsx`) "
    "in the same folder and enter the file name below."
)

default_path = "bulb_sales_sample_1000_rows.xlsx"
file_path = st.sidebar.text_input("Excel file path", value=default_path)

data = load_data(file_path)
data = add_ageing_buckets(data)

# Optional: Region inference if not present
if 'Region' not in data.columns:
    def infer_region(state):
        s = str(state).lower()
        if any(x in s for x in ['west bengal', 'odisha', 'bihar', 'jharkhand', 'assam']):
            return 'East'
        if any(x in s for x in ['maharashtra', 'gujarat', 'rajasthan', 'madhya pradesh']):
            return 'West'
        if any(x in s for x in ['delhi', 'haryana', 'uttar pradesh', 'chandigarh', 'punjab']):
            return 'North'
        if any(x in s for x in ['karnataka', 'tamil nadu', 'telangana', 'kerala', 'andhra pradesh']):
            return 'South'
        return 'Other'
    data['Region'] = data['State'].apply(infer_region)

# ---------- SIDEBAR FILTERS ----------
st.sidebar.subheader("🔍 Filters")

# Date filter (Invoice Date)
min_date = data['Inv_Date'].min()
max_date = data['Inv_Date'].max()
date_range = st.sidebar.date_input(
    "Invoice Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Multi-select filters
def multiselect_filter(label, column):
    options = sorted(data[column].dropna().unique().tolist())
    selected = st.sidebar.multiselect(label, options)
    if selected:
        return data[column].isin(selected)
    return pd.Series(True, index=data.index)

f_region = multiselect_filter("Region", "Region")
f_rsm = multiselect_filter("RSM", "RSM")
f_asm = multiselect_filter("ASM", "ASM")
f_sales = multiselect_filter("Sales Person", "Sales_Person")
f_state = multiselect_filter("State", "State")
f_city = multiselect_filter("City", "City")
f_dealer = multiselect_filter("Dealer", "Name_of_the_dealer")
f_age = multiselect_filter("Ageing Bucket", "Ageing_Bucket")
f_paid = multiselect_filter("Payment Status", "whether_invoice_paid")

# Level selector for performance analysis
st.sidebar.subheader("📊 Performance Analysis Level")
analysis_level = st.sidebar.selectbox(
    "Select Analysis Level",
    options=['State', 'City', 'ASM', 'RSM', 'Dealer'],
    index=0
)
level_column_map = {
    'State': 'State',
    'City': 'City',
    'ASM': 'ASM',
    'RSM': 'RSM',
    'Dealer': 'Name_of_the_dealer'
}
selected_column = level_column_map[analysis_level]

# Apply filters
mask = (
    (data['Inv_Date'].dt.date >= date_range[0]) &
    (data['Inv_Date'].dt.date <= date_range[1]) &
    f_region & f_rsm & f_asm & f_sales & f_state & f_city & f_dealer & f_age & f_paid
)
df = data[mask].copy()

# ---------- METRICS ----------
df['Outstanding_Amount'] = np.where(df['Is_Unpaid'], df['Amount'], 0)

total_outstanding = df['Outstanding_Amount'].sum()
total_amount = df['Amount'].sum()
total_paid = total_amount - total_outstanding

overdue_mask = df['Ageing_Bucket'].isin(['Overdue 0–7 days', 'Overdue 8–14 days', 'Overdue >14 days'])
overdue_outstanding = df.loc[overdue_mask, 'Outstanding_Amount'].sum()

due_0_7 = df.loc[df['Ageing_Bucket'] == 'Due in 0–7 days', 'Outstanding_Amount'].sum()
ovd_0_7 = df.loc[df['Ageing_Bucket'] == 'Overdue 0–7 days', 'Outstanding_Amount'].sum()
ovd_8_14 = df.loc[df['Ageing_Bucket'] == 'Overdue 8–14 days', 'Outstanding_Amount'].sum()

dealers_outstanding = df.groupby('UserID')['Outstanding_Amount'].sum()
high_exposure_dealers = (dealers_outstanding > 1e7).sum()  # >1 crore
perc_overdue = (overdue_outstanding / total_outstanding * 100) if total_outstanding else 0

# ---------- HEADER ----------
st.markdown(
    "<h1 style='margin-bottom:0px;'>💡 Receivables Dashboard</h1>"
    "<p style='color:#666; margin-top:4px;'>Bulb Manufacturing Company – Pan India</p>",
    unsafe_allow_html=True
)

# ---------- KPI CARDS ----------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

def format_inr(x):
    return f"₹{x:,.0f}"

with kpi1:
    st.metric("Total Outstanding", format_inr(total_outstanding))
with kpi2:
    st.metric("Dealers > ₹1 Cr", int(high_exposure_dealers))
with kpi3:
    st.metric("Overdue 0–7 Days", format_inr(ovd_0_7))
with kpi4:
    st.metric("Overdue 8–14 Days", format_inr(ovd_8_14))
with kpi5:
    st.metric("Due in 0–7 Days", format_inr(due_0_7))
with kpi6:
    st.metric("% Outstanding Overdue", f"{perc_overdue:0.1f}%")

st.markdown("---")

# ---------- ROW 1: AGEING & TREND ----------
row1_col1, row1_col2 = st.columns([1.1, 1])

with row1_col1:
    st.subheader("Ageing Bucket Distribution")

    ageing_summary = (
        df.groupby('Ageing_Bucket', as_index=False)['Outstanding_Amount']
        .sum()
        .sort_values('Outstanding_Amount', ascending=False)
    )
    if not ageing_summary.empty:
        fig_age = px.bar(
            ageing_summary,
            x='Ageing_Bucket',
            y='Outstanding_Amount',
            labels={'Outstanding_Amount': 'Outstanding (₹)', 'Ageing_Bucket': 'Bucket'},
            text_auto='.2s'
        )
        fig_age.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with row1_col2:
    st.subheader("Outstanding Trend by Month (Invoice Date)")
    if not df.empty:
        df_month = (
            df.assign(YearMonth=df['Inv_Date'].dt.to_period('M').dt.to_timestamp())
            .groupby('YearMonth', as_index=False)['Outstanding_Amount'].sum()
        )
        fig_trend = px.line(
            df_month,
            x='YearMonth',
            y='Outstanding_Amount',
            markers=True,
            labels={'Outstanding_Amount': 'Outstanding (₹)', 'YearMonth': 'Month'}
        )
        fig_trend.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data for selected filters.")

st.markdown("---")

# ---------- ROW 2: REGION & HIERARCHY ----------
row2_col1, row2_col2 = st.columns([1, 1.2])

with row2_col1:
    st.subheader("Outstanding by Region")
    region_summary = (
        df.groupby('Region', as_index=False)['Outstanding_Amount']
        .sum()
        .sort_values('Outstanding_Amount', ascending=False)
    )
    if not region_summary.empty:
        fig_region = px.pie(
            region_summary,
            names='Region',
            values='Outstanding_Amount',
            hole=0.4
        )
        fig_region.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with row2_col2:
    st.subheader("RSM → ASM → Sales Person (Outstanding)")
    if not df.empty:
        hierarchy_df = (
            df.groupby(['RSM', 'ASM', 'Sales_Person'], as_index=False)['Outstanding_Amount']
            .sum()
            .sort_values('Outstanding_Amount', ascending=False)
        )
        fig_hier = px.treemap(
            hierarchy_df,
            path=['RSM', 'ASM', 'Sales_Person'],
            values='Outstanding_Amount'
        )
        fig_hier.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hier, use_container_width=True)
    else:
        st.info("No data for selected filters.")

st.markdown("---")

# ---------- ROW 2.5: INDIA MAP VISUALIZATION ----------
st.subheader("🗺️ India Sales Performance & Risk Map")
st.markdown("*Interactive map showing state-level revenue and receivables risk analysis*")

if not df.empty:
    map_fig = create_india_map(df)
    if map_fig:
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Add summary statistics below the map
        state_summary = df.groupby('State').agg(
            Total_Revenue=('Amount', 'sum'),
            Outstanding=('Outstanding_Amount', 'sum')
        ).reset_index()
        state_summary['Risk_Pct'] = np.where(
            state_summary['Total_Revenue'] > 0,
            (state_summary['Outstanding'] / state_summary['Total_Revenue']) * 100,
            0
        )
        threshold = state_summary['Risk_Pct'].median()
        high_risk_states = state_summary[state_summary['Risk_Pct'] > threshold]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total States", len(state_summary))
        with col2:
            st.metric("High Risk States", len(high_risk_states))
        with col3:
            st.metric("Median Risk %", f"{threshold:.1f}%")
        with col4:
            top_risk_state = state_summary.loc[state_summary['Risk_Pct'].idxmax(), 'State'] if not state_summary.empty else "N/A"
            st.metric("Highest Risk State", top_risk_state)
    else:
        st.info("No state data available for map visualization.")
else:
    st.info("No data for selected filters.")

st.markdown("---")

# ---------- ROW 3: DEALER TABLE & TOP DEALERS ----------
row3_col1, row3_col2 = st.columns([1.4, 1])

with row3_col2:
    st.subheader("Top 10 Dealers by Outstanding")
    dealer_summary = (
        df.groupby(['UserID', 'Name_of_the_dealer'], as_index=False)['Outstanding_Amount']
        .sum()
        .sort_values('Outstanding_Amount', ascending=False)
        .head(10)
    )
    if not dealer_summary.empty:
        fig_top = px.bar(
            dealer_summary,
            x='Outstanding_Amount',
            y='Name_of_the_dealer',
            orientation='h',
            labels={'Outstanding_Amount': 'Outstanding (₹)', 'Name_of_the_dealer': 'Dealer'},
            text_auto='.2s'
        )
        fig_top.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with row3_col1:
    st.subheader("Dealer Outstanding (Filtered)")
    if not df.empty:
        dealer_table = (
            df.groupby(
                ['UserID', 'Name_of_the_dealer', 'City', 'State', 'ASM', 'Sales_Person'],
                as_index=False
            )[['Outstanding_Amount', 'Amount']]
            .sum()
        )
        dealer_table['> ₹1 Cr Flag'] = np.where(dealer_table['Outstanding_Amount'] > 1e7, '🔴 High', '')
        dealer_table = dealer_table.sort_values('Outstanding_Amount', ascending=False)
        st.dataframe(
            dealer_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data for selected filters.")

st.markdown("---")

# ---------- ROW 4: PERFORMANCE ANALYSIS CHARTS ----------
st.subheader(f"📈 Performance Analysis - {analysis_level} Level")

# Chart 1: Performance Deviation (Z-Score)
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.markdown("#### Performance Deviation (Under vs Over Performing)")
    if not df.empty:
        summary = df.groupby(selected_column)['Amount'].sum().reset_index()
        if len(summary) > 0:
            mean_rev = summary['Amount'].mean()
            std_rev = summary['Amount'].std() if summary['Amount'].std() > 0 else 1
            
            summary['Z_Score'] = (summary['Amount'] - mean_rev) / std_rev
            summary['Color'] = summary['Z_Score'].apply(lambda x: 'red' if x < 0 else 'green')
            summary['Performance'] = summary['Z_Score'].apply(lambda x: 'Underperforming' if x < 0 else 'Overperforming')
            
            # Filter: If too many entities, show Top 10 and Bottom 10 outliers
            if len(summary) > 25:
                top_10 = summary.sort_values('Z_Score', ascending=False).head(10)
                bot_10 = summary.sort_values('Z_Score', ascending=True).head(10)
                chart_data = pd.concat([top_10, bot_10]).sort_values('Z_Score')
            else:
                chart_data = summary.sort_values('Z_Score')
            
            if not chart_data.empty:
                fig_zscore = go.Figure()
                for _, row in chart_data.iterrows():
                    fig_zscore.add_trace(go.Scatter(
                        x=[0, row['Z_Score']],
                        y=[row[selected_column], row[selected_column]],
                        mode='lines+markers',
                        line=dict(color=row['Color'], width=5),
                        marker=dict(size=8),
                        name=row[selected_column],
                        showlegend=False,
                        hovertemplate=f"<b>{row[selected_column]}</b><br>" +
                                     f"Z-Score: {row['Z_Score']:.2f}<br>" +
                                     f"Amount: ₹{row['Amount']:,.0f}<extra></extra>"
                    ))
                
                fig_zscore.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_zscore.update_layout(
                    xaxis_title="Standard Deviations from Average (Left=Under, Right=Over)",
                    yaxis_title=analysis_level,
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode='closest'
                )
                st.plotly_chart(fig_zscore, use_container_width=True)
            else:
                st.info("No data available for this chart.")
        else:
            st.info("No data available for this chart.")
    else:
        st.info("No data for selected filters.")

with row4_col2:
    st.markdown("#### Sales vs Receivables Matrix")
    if not df.empty:
        scatter_data = df.groupby(selected_column).agg(
            Total_Sales=('Amount', 'sum'),
            Total_Receivables=('Outstanding_Amount', 'sum'),
            Invoice_Count=('Inv_No', 'count')
        ).reset_index()
        
        if not scatter_data.empty:
            # Annotate Top 5 outliers (High Receivables)
            outliers = scatter_data.nlargest(5, 'Total_Receivables')
            
            fig_scatter = px.scatter(
                scatter_data,
                x='Total_Sales',
                y='Total_Receivables',
                size='Invoice_Count',
                hover_name=selected_column,
                hover_data={'Total_Sales': ':,.0f', 'Total_Receivables': ':,.0f', 'Invoice_Count': True},
                labels={
                    'Total_Sales': 'Total Revenue Generated (₹)',
                    'Total_Receivables': 'Total Unpaid Amount - Receivables (₹)',
                    'Invoice_Count': 'Number of Invoices'
                },
                title='',
                color='Total_Receivables',
                color_continuous_scale='Purples'
            )
            
            # Add annotations for top 5 outliers
            for _, row in outliers.iterrows():
                fig_scatter.add_annotation(
                    x=row['Total_Sales'],
                    y=row['Total_Receivables'],
                    text=row[selected_column],
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(size=9, color="red", family="Arial Black")
                )
            
            fig_scatter.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No data available for this chart.")
    else:
        st.info("No data for selected filters.")

# Chart 2: Ageing Distribution (Stacked Bar) - using extended ageing buckets
row5_col1, row5_col2 = st.columns(2)

with row5_col1:
    st.markdown("#### Ageing Distribution (Top 10 Riskiest Entities)")
    if not df.empty:
        # Create extended ageing buckets for this chart (similar to Gemini script)
        current_date = df['Inv_Date'].max() if 'Inv_Date' in df.columns else pd.Timestamp(datetime.today().date())
        
        def get_extended_ageing_bucket(row):
            if not row['Is_Unpaid'] or pd.isna(row.get('Due_Date', pd.NaT)):
                return 'Paid'
            delta = (current_date - row['Due_Date']).days
            if delta < 0:
                return 'Not Due Yet'
            elif delta <= 30:
                return '0-30 Days'
            elif delta <= 60:
                return '31-60 Days'
            elif delta <= 90:
                return '61-90 Days'
            else:
                return '>90 Days (Critical)'
        
        df_extended = df.copy()
        df_extended['Extended_Ageing'] = df_extended.apply(get_extended_ageing_bucket, axis=1)
        
        # Pivot for stacked bar
        ageing_pivot = df_extended[df_extended['Is_Unpaid']].pivot_table(
            index=selected_column,
            columns='Extended_Ageing',
            values='Outstanding_Amount',
            aggfunc='sum',
            fill_value=0
        )
        
        ageing_order = ['Not Due Yet', '0-30 Days', '31-60 Days', '61-90 Days', '>90 Days (Critical)']
        present_cols = [c for c in ageing_order if c in ageing_pivot.columns]
        if present_cols:
            ageing_pivot = ageing_pivot[present_cols]
            ageing_pivot['Total_Risk'] = ageing_pivot.sum(axis=1)
            top_risky = ageing_pivot.sort_values('Total_Risk', ascending=False).head(10).drop(columns='Total_Risk')
            
            if not top_risky.empty:
                # Convert to long format for Plotly
                top_risky_reset = top_risky.reset_index()
                top_risky_long = pd.melt(
                    top_risky_reset,
                    id_vars=[selected_column],
                    value_vars=present_cols,
                    var_name='Ageing_Bucket',
                    value_name='Outstanding_Amount'
                )
                
                fig_ageing = px.bar(
                    top_risky_long,
                    x='Outstanding_Amount',
                    y=selected_column,
                    color='Ageing_Bucket',
                    orientation='h',
                    color_discrete_sequence=px.colors.sequential.OrRd,
                    labels={'Outstanding_Amount': 'Outstanding Amount (₹)', selected_column: analysis_level},
                    title=''
                )
                fig_ageing.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(title='Days Overdue')
                )
                st.plotly_chart(fig_ageing, use_container_width=True)
            else:
                st.info("No outstanding debt found.")
        else:
            st.info("No outstanding debt found.")
    else:
        st.info("No data for selected filters.")

with row5_col2:
    st.markdown("#### Sales vs Collections Trend")
    if not df.empty and 'Month_Year' in df.columns:
        # Calculate Sales and Collections by month
        sales_by_month = df.groupby('Month_Year')['Amount'].sum()
        collections_by_month = df[df['Is_Unpaid'] == False].groupby('Month_Year')['Amount'].sum()
        
        trend_data = pd.DataFrame({
            'Month_Year': sales_by_month.index,
            'Sales': sales_by_month.values,
            'Collections': collections_by_month.reindex(sales_by_month.index, fill_value=0).values
        })
        
        if not trend_data.empty:
            fig_trend = go.Figure()
            
            # Sales line
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Month_Year'],
                y=trend_data['Sales'],
                mode='lines+markers',
                name='Total Sales Generated',
                line=dict(color='#2980b9', width=2),
                marker=dict(size=8)
            ))
            
            # Collections line
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Month_Year'],
                y=trend_data['Collections'],
                mode='lines+markers',
                name='Total Collections Received',
                line=dict(color='#27ae60', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Fill between (Receivables gap)
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Month_Year'],
                y=trend_data['Sales'],
                mode='lines',
                name='Gap (Receivables)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_data['Month_Year'],
                y=trend_data['Collections'],
                mode='lines',
                name='Gap',
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_trend.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_trend.update_xaxes(tickangle=45)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No data available for this chart.")
    else:
        st.info("No data for selected filters.")

st.markdown("---")

# ---------- ROW 5: INVOICE DETAIL ----------
st.subheader("Invoice Detail (Filtered Context)")
if not df.empty:
    detail_cols = [
        'Inv_No', 'Inv_Date', 'Due_Date', 'Credit_Period', 'Amount',
        'Outstanding_Amount', 'Ageing_Bucket', 'Days_To_Due', 'Days_Overdue',
        'whether_invoice_paid', 'Name_of_the_dealer', 'City', 'Sales_Person', 'Phone_Number'
    ]
    show_cols = [c for c in detail_cols if c in df.columns]
    detail_df = df[show_cols].sort_values('Due_Date', ascending=True)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
else:
    st.info("No data for selected filters.")
