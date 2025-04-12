
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")  # Replace with your key

st.set_page_config(page_title="AI Survey Analyzer", layout="wide")
st.title("🧠 AI-Powered Survey Analyzer")

uploaded_file = st.file_uploader("📤 Upload your survey CSV file", type=["csv"])

def detect_column_types(df):
    types = {"open_ended": [], "likert": [], "ratings": [], "checkbox": [], "matrix": []}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if df[col].str.split().str.len().mean() > 3:
                types["open_ended"].append(col)
            elif len(unique_vals) <= 5:
                types["likert"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].max() <= 10:
                types["ratings"].append(col)
            else:
                types["matrix"].append(col)
        elif df[col].dropna().isin(['Yes', 'No', 'Checked', 'Unchecked']).any():
            types["checkbox"].append(col)
    return types

def generate_wordcloud(text_series):
    text = " ".join(text_series.dropna().astype(str))
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z ]", "", text)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    st.caption('🔍 Interpretation: This chart shows the most frequent words mentioned.')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    column_types = detect_column_types(df)

    st.subheader("🧪 Suggested Analyses")

    if column_types["open_ended"]:
        if st.button("📝 Analyze Open-Ended Feedback"):
            generate_wordcloud(df[column_types["open_ended"][0]])

    if column_types["likert"]:
        if st.button("📊 Visualize Likert Scale Averages"):
            likert_col = column_types["likert"][0]
            st.bar_chart(df[likert_col].value_counts())
            st.caption('🔍 Distribution of responses across Likert options.')

    if column_types["ratings"]:
        if st.button("⭐ Show Average Ratings"):
            ratings = df[column_types["ratings"]].mean().sort_values(ascending=False)
            st.bar_chart(ratings)
            st.caption('🔍 Average scores for each rated item.')

    if column_types["checkbox"]:
        if st.button("☑️ Analyze Checkbox Selections"):
            checkbox_col = column_types["checkbox"][0]
            counts = df[checkbox_col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_title(f"Responses for {checkbox_col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.caption('🔍 Checkbox selection counts across responses.')

    if column_types["matrix"]:
        if st.button("🧮 Matrix Summary (Average by Row)"):
            matrix_df = df[column_types["matrix"]]
            st.dataframe(matrix_df.mean(axis=1).to_frame(name="Row Average"))
            st.caption('🔍 These are average values per row from matrix questions.')

    st.subheader("📊 Compare Results by Segment")
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    if numeric_columns and categorical_columns:
        metric_col = st.selectbox("Choose a numeric column to compare", numeric_columns)
        segment_col = st.selectbox("Choose a segment column (e.g. Gender, Age Group)", categorical_columns)

        if metric_col and segment_col:
            st.write(f"### {metric_col} by {segment_col}")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=df, x=segment_col, y=metric_col, ci=None, ax=ax)
            ax.set_title(f"Average {metric_col} by {segment_col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.caption(f'🔍 Average {metric_col} values for each {segment_col} group.')

    st.subheader("🧭 Segment Explorer: Analyze All Columns by Segment")
    segment_column = st.selectbox("Choose a segment column to explore all other data by:", categorical_columns, key="segment_explorer")

    if segment_column:
        st.markdown(f"Analyzing responses by **{segment_column}**...")
        for col in df.columns:
            if col == segment_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                st.write(f"#### 📈 {col} by {segment_column}")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(data=df, x=segment_column, y=col, ci=None, ax=ax)
                ax.set_title(f"{col} by {segment_column}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.caption(f'🔍 This chart shows how {col} differs across {segment_column}.')

            elif df[col].dtype == 'object' and df[col].nunique() < 10:
                st.write(f"#### 📊 Distribution of {col} by {segment_column}")
                crosstab = pd.crosstab(df[segment_column], df[col], normalize='index') * 100
                st.bar_chart(crosstab)
                st.caption(f'🔍 This chart shows % distribution of {col} by {segment_column}.')

    st.subheader("💬 Ask a Question (Natural Language with GPT)")
    user_query = st.text_input("Type your question about the survey data:")
    if user_query:
        try:
            llm = OpenAI(api_token=OPENAI_API_KEY)
            sdf = SmartDataframe(df, config={"llm": llm})
            with st.spinner("Thinking..."):
                result = sdf.chat(user_query)
            st.success("✅ Here's what I found:")
            st.write(result)
        except Exception as e:
            st.error("❌ Error processing query. Check your OpenAI API key or question format.")
            st.text(str(e))

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())
