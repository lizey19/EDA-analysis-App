import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Title and Description
st.title("🚀 Advanced & Interactive EDA Dashboard")
st.markdown("""
This app provides **advanced statistical analysis** and **interactive EDA visualizations**. 
Upload your dataset to explore insights, correlations, and distributions in a unique and intuitive way.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    # Dataset info
    st.subheader("📊 Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes)

    # Missing values
    st.subheader("⚠️ Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all'))

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Interactive correlation heatmap
        fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Column analysis
    st.subheader("📌 Column Analysis")
    column = st.selectbox("Select a column", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}")
        st.write(df[column].describe())

        # Histogram + KDE
        fig = px.histogram(df, x=column, marginal="box", nbins=30, title=f"Distribution of {column}", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

        # Violin plot
        fig_violin = px.violin(df, y=column, box=True, points="all", title=f"Violin Plot of {column}")
        st.plotly_chart(fig_violin, use_container_width=True)

    else:
        st.write(f"Value counts of {column}")
        st.write(df[column].value_counts())

        # Interactive bar chart
        fig_bar = px.bar(df[column].value_counts().reset_index(),
                        x="index", y=column, title=f"Count of {column}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pairplot
    if not numeric_df.empty:
        st.subheader("🔗 Pairwise Relationships (Numeric Columns)")
        fig_pair = px.scatter_matrix(numeric_df, dimensions=numeric_df.columns, title="Scatter Matrix")
        st.plotly_chart(fig_pair, use_container_width=True)

    # Outlier detection
    st.subheader("🚨 Outlier Detection (IQR Method)")
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        st.write(f"{col}: {len(outliers)} outliers detected")

    # Interactive 3D plot
    if len(numeric_df.columns) >= 3:
        st.subheader("🌐 3D Visualization")
        cols_3d = st.multiselect("Select 3 numeric columns for 3D scatter plot", numeric_df.columns, numeric_df.columns[:3])
        if len(cols_3d) == 3:
            fig_3d = px.scatter_3d(df, x=cols_3d[0], y=cols_3d[1], z=cols_3d[2], color=cols_3d[0], title="3D Scatter Plot")
            st.plotly_chart(fig_3d, use_container_width=True)
