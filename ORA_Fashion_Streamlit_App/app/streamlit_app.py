"""
ORA Fashion Customer Intelligence Platform
Streamlit Application

Run locally:
    streamlit run app/streamlit_app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from pathlib import Path

st.set_page_config(
    page_title="ORA Fashion Customer Intelligence",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ ORA Fashion Customer Intelligence Platform")
st.caption("Customer Segmentation | CLV Analysis | Churn Risk | Business Intelligence")

st.sidebar.header("Upload Data")
customers_file = st.sidebar.file_uploader("Upload Customers Excel file", type=["xlsx"])
orders_file = st.sidebar.file_uploader("Upload Orders Excel file", type=["xlsx"])

@st.cache_data
def load_data(customers_file, orders_file):
    customers = pd.read_excel(customers_file)
    orders = pd.read_excel(orders_file)
    if "@" in customers.columns:
        customers = customers.drop(columns=["@"])
    orders["Date"] = pd.to_datetime(orders["Date"], errors="coerce")
    merged = orders.merge(customers, on="Customer_ID", how="left")
    return customers, orders, merged

def build_customer_features(orders, customers):
    snapshot_date = orders["Date"].max() + pd.Timedelta(days=1)

    features = orders.groupby("Customer_ID").agg(
        First_Purchase=("Date", "min"),
        Last_Purchase=("Date", "max"),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Sales_Amount", "sum"),
        Quantity=("Quantity", "sum")
    ).reset_index()

    features["Recency"] = (snapshot_date - features["Last_Purchase"]).dt.days
    features["Tenure_Days"] = (features["Last_Purchase"] - features["First_Purchase"]).dt.days + 1
    features["Average_Order_Value"] = features["Monetary"] / features["Frequency"]
    features["Churn_Status"] = np.where(features["Recency"] > 183, "Churned", "Active")

    churn_rate = (features["Churn_Status"] == "Churned").mean()
    lifetime_multiplier = min(1 / max(churn_rate, 0.01), 3)
    features["CLV"] = features["Average_Order_Value"] * features["Frequency"] * lifetime_multiplier

    features["Churn_Risk"] = pd.cut(
        features["Recency"],
        bins=[-1, 90, 183, 10000],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    features = features.merge(customers, on="Customer_ID", how="left")
    return features

if customers_file is None or orders_file is None:
    st.info("Upload the Customers and Orders Excel files to start the analysis.")
    st.markdown("""
    ### Expected files
    - **Customers.xlsx** with `Customer_ID`, `GENDER`, `AGE`, `GEOGRAPHY`
    - **Orders.xlsx** with `Date`, `Customer_ID`, `Transaction_ID`, `SKU_Category`, `SKU`, `Quantity`, `Sales_Amount`
    """)
    st.stop()

customers, orders, merged = load_data(customers_file, orders_file)
features = build_customer_features(orders, customers)

total_revenue = orders["Sales_Amount"].sum()
total_customers = customers["Customer_ID"].nunique()
total_transactions = orders["Transaction_ID"].nunique()
avg_order_value = total_revenue / total_transactions
churn_rate = (features["Churn_Status"] == "Churned").mean()
avg_clv = features["CLV"].mean()

st.subheader("Executive KPIs")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"€{total_revenue:,.0f}")
c2.metric("Customers", f"{total_customers:,}")
c3.metric("Transactions", f"{total_transactions:,}")
c4.metric("Avg Order Value", f"€{avg_order_value:,.0f}")
c5.metric("Churn Rate", f"{churn_rate:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Executive Overview", "Customer Analytics", "CLV", "Churn Risk"])

with tab1:
    st.subheader("Revenue Trend")
    monthly = orders.assign(Month=orders["Date"].dt.to_period("M").astype(str)).groupby("Month")["Sales_Amount"].sum().reset_index()
    st.plotly_chart(px.line(monthly, x="Month", y="Sales_Amount", markers=True, title="Monthly Revenue Trend"), use_container_width=True)

    st.subheader("Revenue by Geography")
    geo = merged.groupby("GEOGRAPHY")["Sales_Amount"].sum().reset_index().sort_values("Sales_Amount", ascending=False)
    st.plotly_chart(px.bar(geo, x="GEOGRAPHY", y="Sales_Amount", title="Revenue by Geography"), use_container_width=True)

with tab2:
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = customers["GENDER"].value_counts().reset_index()
        gender.columns = ["Gender", "Customers"]
        st.plotly_chart(px.pie(gender, names="Gender", values="Customers", title="Gender Distribution"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(customers, x="AGE", nbins=25, title="Age Distribution"), use_container_width=True)

    st.subheader("Product Category Revenue")
    cat = orders.groupby("SKU_Category")["Sales_Amount"].sum().reset_index().sort_values("Sales_Amount", ascending=False).head(15)
    st.plotly_chart(px.bar(cat, x="SKU_Category", y="Sales_Amount", title="Top Product Categories"), use_container_width=True)

with tab3:
    st.subheader("Customer Lifetime Value")
    st.metric("Average CLV", f"€{avg_clv:,.0f}")
    st.plotly_chart(px.histogram(features, x="CLV", nbins=40, title="CLV Distribution"), use_container_width=True)

    st.subheader("Top 20 Customers by CLV")
    st.dataframe(features.sort_values("CLV", ascending=False).head(20), use_container_width=True)

with tab4:
    st.subheader("Churn Risk")
    risk = features["Churn_Risk"].value_counts().reset_index()
    risk.columns = ["Risk", "Customers"]
    st.plotly_chart(px.bar(risk, x="Risk", y="Customers", title="Customers by Churn Risk"), use_container_width=True)

    selected_risk = st.selectbox("Filter customers by risk", ["All"] + list(features["Churn_Risk"].astype(str).unique()))
    display = features.copy()
    if selected_risk != "All":
        display = display[display["Churn_Risk"].astype(str) == selected_risk]
    st.dataframe(display.sort_values("Recency", ascending=False), use_container_width=True)

st.download_button(
    "Download Customer Intelligence Output",
    data=features.to_csv(index=False),
    file_name="ora_fashion_customer_intelligence_output.csv",
    mime="text/csv"
)
