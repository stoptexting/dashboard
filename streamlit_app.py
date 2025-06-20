import streamlit as st
import pandas as pd
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide"
)

# Required columns in each data file
REQUIRED_COLS = {"Date", "Email", "Country Name", "Amount Net", "Order ID"}

@st.cache_data
def load_and_validate(file) -> pd.DataFrame:
    try:
        if file.name.lower().endswith(("xlsx", "xls")):
            df = pd.read_excel(file)
        elif file.name.lower().endswith("csv"):
            df = pd.read_csv(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return None
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"Invalid format in {file.name}. Missing columns: {', '.join(missing)}")
        return None
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Amount'] = pd.to_numeric(df['Amount Net'], errors='coerce')
    df = df.dropna(subset=['Amount'])
    df['Country'] = df['Country Name']
    df['CustomerID'] = df['Email']
    return df

# --- Upload Section ---
st.title("Welcome to the Sales Analytics Dashboard")
st.markdown("Drag & drop Excel/CSV files with your sales data below.")
uploaded = st.file_uploader(
    "Upload sales data:", type=['xlsx','xls','csv'], accept_multiple_files=True
)
if not uploaded:
    st.info("Awaiting file upload...")
    st.stop()
dfs = []
for file in uploaded:
    df = load_and_validate(file)
    if df is not None:
        dfs.append(df)
if not dfs:
    st.error("No valid files uploaded. Please check formats.")
    st.stop()
# Concatenate and extract time info
data = pd.concat(dfs, ignore_index=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.to_period('M').dt.to_timestamp()

# --- Tabs ---
tabs = st.tabs(["Overview","Sales Performance","Customer Behavior","Funnel Analysis","Segmentation","Forecasting"])

# Overview Tab
with tabs[0]:
    st.header("Global Overview")
    year_sel = st.selectbox("Select Year", sorted(data['Year'].unique(), reverse=True), key='ov_year')
    df_year = data[data['Year']==year_sel]
    # Sales heatmap
    sales_country = df_year.groupby('Country')['Amount'].sum().reset_index()
    fig_hi = px.choropleth(sales_country, locations='Country', locationmode='country names', color='Amount', title=f"Sales by Country in {year_sel}")
    st.plotly_chart(fig_hi, use_container_width=True, key='overview_heatmap')
    # Metrics with tooltips
    curr_rev = df_year['Amount'].sum()
    prev_rev = data[data['Year']==year_sel-1]['Amount'].sum() if (year_sel-1) in data['Year'].unique() else 0
    delta_rev = curr_rev - prev_rev
    unique_cust = df_year['CustomerID'].nunique()
    total_orders = df_year.shape[0]
    recurring = data[data['Year']==year_sel].groupby('CustomerID').size().gt(1).sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        has_prev = (year_sel - 1) in data['Year'].unique()
        if has_prev:
            st.metric("Revenue", f"${curr_rev:,.0f}", delta=round(delta_rev,2), delta_color="normal")
        else:
            st.metric("Revenue", f"${curr_rev:,.0f}")
        st.caption("ℹ️ Total sales revenue for the selected year.")
    # Unique Customers with YoY
    prev_unique = data[data['Year']==year_sel-1]['CustomerID'].nunique() if (year_sel-1) in data['Year'].unique() else 0
    with c2:
        st.metric(
            "Unique Customers", unique_cust,
            delta=int(unique_cust - prev_unique),
            delta_color="normal"
        )
        st.caption("ℹ️ Number of distinct customers who made at least one purchase (YoY delta).")
    # Total Orders with YoY
    prev_orders = data[data['Year']==year_sel-1].shape[0] if (year_sel-1) in data['Year'].unique() else 0
    with c3:
        st.metric(
            "Total Orders", total_orders,
            delta=int(total_orders - prev_orders),
            delta_color="normal"
        )
        st.caption("ℹ️ Total number of orders placed in the selected year (YoY delta).")
    # Recurring Customers with YoY
    prev_recurring = data[data['Year']==year_sel-1].groupby('CustomerID').size().gt(1).sum() if (year_sel-1) in data['Year'].unique() else 0
    with c4:
        st.metric(
            "Recurring Customers", recurring,
            delta=int(recurring - prev_recurring),
            delta_color="normal"
        )
        st.caption("ℹ️ Customers who placed more than one order this year (YoY delta).")
    # Payment Processor Fees (if available) (if available)
    if {'Processor','Fee'}.issubset(data.columns):
        pcol1, pcol2 = st.columns(2)
        # Sum of PayPal and Stripe fees
        pp_fee = df_year[df_year['Processor'].isin(['PayPal','Stripe'])]['Fee'].sum()
        payhip_fee = df_year[df_year['Processor']=='Payhip']['Fee'].sum()
        pcol1.metric("PayPal/Stripe Fees", f"${pp_fee:,.2f}")
        pcol1.caption("ℹ️ Total fees paid to PayPal and Stripe in the selected year.")
        pcol2.metric("Payhip Fees", f"${payhip_fee:,.2f}")
        pcol2.caption("ℹ️ Total fees paid to Payhip in the selected year.")
    else:
        st.info("Payment fee data not available.")

# Sales Performance Tab
with tabs[1]:
    st.header("Sales Performance KPIs")
    # Filters
    year_perf = st.selectbox("Year", sorted(data['Year'].unique(), reverse=True), key='sp_year')
    df_sp = data[data['Year']==year_perf]
    months = sorted(df_sp['Month'].dt.strftime('%Y-%m').unique().tolist())
    month_sel = st.selectbox("Month", ["All"]+months, key='sp_month')
    if month_sel != "All":
        df_sp = df_sp[df_sp['Month'].dt.strftime('%Y-%m')==month_sel]
    # Revenue & growth
    rev_curr = df_sp['Amount'].sum()
    if month_sel=="All":
        rev_prev = data[data['Year']==year_perf-1]['Amount'].sum() if (year_perf-1) in data['Year'].unique() else 0
    else:
        prev_m = (pd.to_datetime(month_sel)+pd.offsets.MonthEnd(-1)).to_period('M').to_timestamp()
        rev_prev = data[data['Month']==prev_m]['Amount'].sum()
    rev_delta = rev_curr-rev_prev
    st.metric("Revenue", f"${rev_curr:,.0f}", delta=round(rev_delta,2), delta_color="normal")
    # AOV
    orders = df_sp.shape[0]
    aov = rev_curr / orders if orders > 0 else 0
    st.metric("Avg Order Value", f"${aov:,.2f}")
    # Payment Processor Fees (Annual)
    # Payment Processor Fees (Annual)
    if {'PayPal/Stripe Fee', 'Payhip Fee'}.issubset(df_sp.columns):
        fee_col1, fee_col2 = st.columns(2)
        pp_stripe_fee = df_sp['PayPal/Stripe Fee'].sum()
        payhip_fee = df_sp['Payhip Fee'].sum()
        with fee_col1:
            st.metric("PayPal/Stripe Fees", f"${pp_stripe_fee:,.2f}")
            st.caption("ℹ️ Total PayPal & Stripe fees for the selected period.")
        with fee_col2:
            st.metric("Payhip Fees", f"${payhip_fee:,.2f}")
            st.caption("ℹ️ Total Payhip fees for the selected period.")
    else:
        st.info("Payment fee data not available for this period.")
    # Trend chart
    trend = data[data['Year']==year_perf].groupby('Month')['Amount'].sum().reset_index()
    fig_tr = px.line(trend, x='Month', y='Amount', title=f"Monthly Revenue in {year_perf}")
    st.plotly_chart(fig_tr, use_container_width=True, key='trend_line')
    # Top segments
    # By Country
    top_c = df_sp.groupby('Country')['Amount'].sum().nlargest(10).reset_index()
    fig_tc = px.bar(top_c, x='Country', y='Amount', title="Top 10 Countries by Revenue")
    st.plotly_chart(fig_tc, use_container_width=True, key='top_countries')
    # (Removed school segment from Sales Performance) of the Year 🎉
    top_sp = df_sp.groupby('CustomerID')['Amount'].sum().nlargest(1)
    if not top_sp.empty:
        spender = top_sp.index[0]
        spend_amt = top_sp.iloc[0]
        st.subheader(f"Top Spender of {year_perf} 🎉")
        st.write(f"**Customer:** {spender}")
        st.write(f"**Total Spent:** ${spend_amt:,.2f}")

# Customer Behavior Tab
with tabs[2]:
    st.header("Customer Behavior KPIs")
    # Year filter for behavior analysis
    year_cb = st.selectbox(
        "Select Year for Customer Behavior", 
        sorted(data['Year'].unique(), reverse=True), 
        key='cb_year'
    )
    df_cb = data[data['Year'] == year_cb]

    # Repeat Purchase Rate with YoY delta
    cust_orders = df_cb.groupby('CustomerID').size()
    repeat_curr = cust_orders[cust_orders > 1].count() / cust_orders.count() if cust_orders.count() else 0
    # previous year
    if (year_cb-1) in data['Year'].unique():
        df_prev_cb = data[data['Year'] == year_cb-1]
        prev_orders = df_prev_cb.groupby('CustomerID').size()
        repeat_prev = prev_orders[prev_orders > 1].count() / prev_orders.count() if prev_orders.count() else 0
    else:
        repeat_prev = None
    if repeat_prev is not None:
        st.metric("Repeat Purchase Rate", f"{repeat_curr:.2%}", delta=round(repeat_curr - repeat_prev,2), delta_color="normal")
    else:
        st.metric("Repeat Purchase Rate", f"{repeat_curr:.2%}")
    st.caption("ℹ️ % of customers who made more than one purchase in the selected year.")

    # Avg Orders per Customer with YoY delta
    avg_curr = cust_orders.mean() if not cust_orders.empty else 0
    if (year_cb-1) in data['Year'].unique():
        prev_avg = prev_orders.mean() if not prev_orders.empty else 0
    else:
        prev_avg = None
    if prev_avg is not None:
        st.metric("Avg Orders per Customer", f"{avg_curr:.2f}", delta=round(avg_curr - prev_avg,2), delta_color="normal")
    else:
        st.metric("Avg Orders per Customer", f"{avg_curr:.2f}")
    st.caption("ℹ️ Average number of orders per customer in the selected year.")

    # High-Value Customers with YoY delta
    clv = df_cb.groupby('CustomerID')['Amount'].sum()
    clv_nonzero = clv[clv > 0]
    if not clv_nonzero.empty:
        threshold = clv_nonzero.quantile(0.9)
        hv_curr = (clv >= threshold).sum() / clv.count()
    else:
        threshold = 0
        hv_curr = 0
    # previous HV
    if (year_cb-1) in data['Year'].unique():
        clv_prev = data[data['Year']==year_cb-1].groupby('CustomerID')['Amount'].sum()
        clv_prev_nonzero = clv_prev[clv_prev > 0]
        if not clv_prev_nonzero.empty:
            thr_prev = clv_prev_nonzero.quantile(0.9)
            hv_prev = (clv_prev >= thr_prev).sum() / clv_prev.count()
        else:
            hv_prev = None
    else:
        hv_prev = None
    if hv_prev is not None:
        st.metric(
            label=f"High-Value Customers (≥ ${threshold:,.2f})",
            value=f"{hv_curr:.2%}",
            delta=round(hv_curr - hv_prev,2),
            delta_color="normal"
        )
    else:
        st.metric(
            label=f"High-Value Customers (≥ ${threshold:,.2f})",
            value=f"{hv_curr:.2%}"
        )
    st.caption(f"ℹ️ % of customers whose total spend is in the top 10% (threshold ≥ ${threshold:,.2f}).")

    # Purchase Frequency Distribution
    freq_df = cust_orders.reset_index(name='Orders')
    dist = freq_df['Orders'].value_counts().reset_index()
    dist.columns = ['Orders', 'Customers']
    dist = dist.sort_values('Orders')
    fig_pf = px.bar(
        dist,
        x='Orders',
        y='Customers',
        title="Purchase Frequency Distribution"
    )
    st.plotly_chart(fig_pf, use_container_width=True, key='freq_dist')

# Funnel Analysis Tab
with tabs[3]:
    # No funnel data available for this dashboard
    st.header("Funnel & Conversion Analysis")
    st.write("Funnel data not provided. Please ensure lead and conversion data is available to populate this section.")

# Segmentation Tab
with tabs[4]:
    st.header("Customer Segmentation")
    # Year filter for segmentation
    year_seg = st.selectbox(
        "Select Year for Segmentation",
        sorted(data['Year'].unique(), reverse=True),
        key='seg_year'
    )
    df_seg = data[data['Year'] == year_seg]

    # --- RFM Segmentation ---
    import datetime as dt
    snapshot = df_seg['Date'].max() + dt.timedelta(days=1)
    rfm = df_seg.groupby('CustomerID').agg(
        Recency=('Date', lambda x: (snapshot - x.max()).days),
        Frequency=('Order ID', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    # Define RFM labels
    r_labels = [5,4,3,2,1]
    f_labels = [1,2,3,4,5]
    m_labels = [1,2,3,4,5]
    # Score RFM with duplicates dropped
    # Use qcut without labels and adjust codes
    try:
        r_bins = pd.qcut(rfm['Recency'], 5, duplicates='drop')
        rfm['R_Score'] = r_bins.cat.codes + 1
    except Exception:
        rfm['R_Score'] = 3
    try:
        f_bins = pd.qcut(rfm['Frequency'], 5, duplicates='drop')
        rfm['F_Score'] = f_bins.cat.codes + 1
    except Exception:
        rfm['F_Score'] = 3
    try:
        m_bins = pd.qcut(rfm['Monetary'], 5, duplicates='drop')
        rfm['M_Score'] = m_bins.cat.codes + 1
    except Exception:
        rfm['M_Score'] = 3
    # Sum RFM scores
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

    # RFM Scatter plot
    fig_rfm = px.scatter(
        rfm,
        x='Recency', y='Frequency',
        size='Monetary',
        color='RFM_Score',
        hover_data=['CustomerID'],
        title="RFM Segmentation Scatter"
    )
    st.plotly_chart(fig_rfm, use_container_width=True, key='rfm_scatter')

    st.markdown(
        """
**How to read the RFM Segmentation Scatter Plot:**

- **X‑axis (Recency):** Number of days since a customer’s last purchase. Closer to 0 (left) indicates more recent activity.
- **Y‑axis (Frequency):** Total number of orders placed by the customer in the period. Higher values (up) indicate more frequent purchases.
- **Marker size (Monetary):** Total amount spent. Larger bubbles represent higher spenders.
- **Marker color (RFM Score):** Combined R+F+M score (5 highest, 3 moderate, 1 lowest), with darker colors highlighting your best customers.

Use this chart to identify your “Champions” (high frequency, recent, high spend) in the upper-left quadrant with large, dark bubbles.
        """,
        unsafe_allow_html=True
    )

    

# Forecasting Tab
with tabs[5]:
    st.header("Sales Forecasting")
    st.warning("Experimental Feature: forecasts may be inaccurate. Use results for guidance only.")
    # Aggregate monthly sales
    monthly = data.groupby('Month')['Amount'].sum().reset_index()
    st.subheader("Historical Monthly Revenue")
    fig_hist = px.line(
        monthly, x='Month', y='Amount', title="Historical Revenue by Month"
    )
    st.plotly_chart(fig_hist, use_container_width=True, key='hist_revenue')

    # Simple Exponential Smoothing Forecast
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        # Fit model on historical data
        model = ExponentialSmoothing(
            monthly['Amount'], trend='add', seasonal=None, initialization_method='estimated'
        ).fit()
        # Forecast next 6 months
        forecast_periods = 6
        forecast = model.forecast(forecast_periods)
        # Build forecast DataFrame
        last_date = monthly['Month'].max()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=forecast_periods, freq='MS')
        df_forecast = pd.DataFrame({'Month': future_dates, 'Forecast': forecast.values})
        forecast_type = 'Exponential Smoothing'
    except ImportError:
        st.warning("statsmodels not installed, using simple moving average forecast instead.")
        # Fallback: simple 6-month moving average of last 12 months
        window = 12 if len(monthly) >= 12 else len(monthly)
        rolling_mean = monthly['Amount'].rolling(window).mean().iloc[-1]
        forecast_periods = 6
        last_date = monthly['Month'].max()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=forecast_periods, freq='MS')
        df_forecast = pd.DataFrame({'Month': future_dates, 'Forecast': [rolling_mean]*forecast_periods})
        forecast_type = 'Moving Average'
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        st.stop()
    # Combine historical and forecast
    df_combined = pd.concat([
        monthly.rename(columns={'Amount': 'Value'}).assign(Type='Historical'),
        df_forecast.rename(columns={'Forecast': 'Value'}).assign(Type=forecast_type)
    ])
    st.subheader(f"6-Month Revenue Forecast ({forecast_type})")
    fig_fore = px.line(
        df_combined, x='Month', y='Value', color='Type',
        title="Historical and Forecasted Revenue"
    )
    st.plotly_chart(fig_fore, use_container_width=True, key='forecast_revenue')

    # Forecast summary
    total_fore = df_forecast['Forecast'].sum()
    st.metric("Total Forecasted Revenue (6 months)", f"${total_fore:,.0f}")
    st.caption("ℹ️ Sum of forecasted monthly revenues for the next 6 months.")
