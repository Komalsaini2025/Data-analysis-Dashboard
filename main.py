import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto Data Dashboard", layout="wide")
sns.set_style("whitegrid")

# ---------------- OOP Class ----------------
class AutoDashboard:
    def __init__(self, df):
        self.df = df
        self.cleaned_df = None

    def clean_data(self):
        df = self.df.copy()

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        # Drop fully empty columns
        df.dropna(axis=1, how="all", inplace=True)

        # Fill numeric NaN with median
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical NaN
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("Unknown")

        self.cleaned_df = df

        # Identify date column
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
            self.date_col = date_cols[0]
        else:
            self.date_col = None

        # Identify numeric and categorical columns
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        return df

    def summary(self):
        return self.cleaned_df.describe()

    def correlation(self):
        if len(self.numeric_cols) > 1:
            return self.cleaned_df[self.numeric_cols].corr()
        return None


# ---------------- Streamlit UI ----------------
st.title("📊 Universal Auto Data Dashboard (Enhanced Version)")

uploaded_file = st.file_uploader("Upload ANY CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    dashboard = AutoDashboard(df)
    
    st.subheader("🔍 Original Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # ----------- Cleaning ----------- 
    st.subheader("🧹 Data Cleaning")
    cleaned_df = dashboard.clean_data()
    st.success("Data cleaning completed.")
    st.dataframe(cleaned_df, use_container_width=True)

    # ----------- Basic Stats ----------- 
    st.subheader("📌 Basic Statistics (Cleaned Data)")
    st.dataframe(dashboard.summary(), use_container_width=True)

    # ----------- Correlation Heatmap ----------- 
    st.subheader("🔥 Correlation Heatmap")
    corr = dashboard.correlation()
    if corr is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # ----------- Trend Line (Auto Date) ----------- 
    if dashboard.date_col and len(dashboard.numeric_cols) > 0:
        st.subheader("📉 Trend Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(cleaned_df[dashboard.date_col], cleaned_df[dashboard.numeric_cols[0]], linewidth=2)
        ax2.set_title(f"Trend of {dashboard.numeric_cols[0]}")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # ----------- Bar Chart ----------- 
    st.subheader("📊 Automatic Bar Chart")
    if len(dashboard.cat_cols) > 0 and len(dashboard.numeric_cols) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        cleaned_df.groupby(dashboard.cat_cols[0])[dashboard.numeric_cols[0]].sum().plot(kind='bar', ax=ax3)
        ax3.set_title(f"{dashboard.numeric_cols[0]} by {dashboard.cat_cols[0]}")
        st.pyplot(fig3)
    else:
        st.info("Not enough category & numeric columns for bar chart.")

    # ----------- Histogram ----------- 
    st.subheader("📦 Numeric Distribution (Histogram)")
    if len(dashboard.numeric_cols) > 0:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.histplot(cleaned_df[dashboard.numeric_cols[0]], bins=30, kde=True, ax=ax4)
        ax4.set_title(f"Distribution of {dashboard.numeric_cols[0]}")
        st.pyplot(fig4)

    # ----------- Pie Chart (Auto – Fixed) ----------- 
    st.subheader("🥧 Pie Chart (Auto Selected Category)")

    if len(dashboard.cat_cols) > 0:

        # 1️⃣ Choose best column for pie chart
        suitable_cols = [col for col in dashboard.cat_cols if cleaned_df[col].nunique() <= 15]

        # If none found, use column with lowest unique values
        if len(suitable_cols) == 0:
            pie_col = cleaned_df[dashboard.cat_cols].nunique().idxmin()
        else:
            pie_col = suitable_cols[0]

        # 2️⃣ Generate Pie Chart
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        cleaned_df[pie_col].value_counts().head(10).plot(
            kind="pie",
            autopct="%1.1f%%",
            shadow=True,
            ax=ax5
        )
        ax5.set_title(f"Top Categories of '{pie_col}'")
        ax5.set_ylabel("")
        st.pyplot(fig5)

    else:
        st.info("No categorical column available for pie chart.")

else:
    st.warning("Upload a CSV file to generate dashboard.")
