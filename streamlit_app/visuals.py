import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def display_visuals(df):
    # Gender Distribution
    st.markdown("### 👨‍⚕️ Gender Distribution")
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax1)
        ax1.set_ylabel("Count")
        ax1.set_title("Gender Distribution")
        st.pyplot(fig1)
    else:
        st.info("Gender column not found.")

    # Age Distribution
    st.markdown("### 📊 Age Distribution")
    if 'Age' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax2)
        ax2.set_title("Age Distribution")
        st.pyplot(fig2)
    else:
        st.info("Age column not found.")

    # Correlation Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    try:
        numeric_df = df.select_dtypes(include='number')
        fig3, ax3 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Error displaying heatmap: {e}")
