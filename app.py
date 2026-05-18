import streamlit as st
import pandas as pd
import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def analyze_dataframe(df):
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "numeric_summary": df.describe().round(2).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records")
    }

    prompt = f"""You are a senior data analyst. Analyze this dataset and return ONLY a JSON object
with exactly these keys (no other text before or after the JSON):

{{
  "headline": "one punchy sentence summarizing the most important thing about this data",
  "key_insights": ["insight 1 with specific numbers", "insight 2", "insight 3", "insight 4"],
  "anomalies": ["any data quality issue or outlier you notice, or 'No anomalies detected'"],
  "business_narrative": "Write 2-3 paragraphs explaining what story this data tells, as if presenting to a business audience.",
  "recommended_actions": ["action 1", "action 2", "action 3"]
}}

Dataset information:
{json.dumps(stats, default=str, indent=2)}"""
    
    response = client.messages.create(
                model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

st.set_page_config(page_title="Data Narrator", page_icon="chart_with_upwards_trend", layout="wide")

st.title("DataNarrator")
st.caption("Upload a CSV · Get instant AI-powered business insights · Built by Rucha Joshi")
st.divider()

uploaded_file = st.file_uploader(
    "Upload your CSV file to get started",
    type=["csv"],
    help="Any CSV file works — sales data, HR data, financial data, anything"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Numeric columns", len(df.select_dtypes(include='number').columns))
    col4.metric("Missing values", f"{df.isnull().sum().sum():,}")

    st.subheader("Preview of your data")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()
    if st.button("Generate AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Claude is reading your data and writing insights..."):
            try:
                result = analyze_dataframe(df)

                st.subheader(result["headline"])
                st.write(result["business_narrative"])

                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("Key insights")
                    for insight in result["key_insights"]:
                        st.markdown(f"- {insight}")

                    st.subheader("Recommended actions")
                    for action in result["recommended_actions"]:
                        st.info(action)

                with col_b:
                    st.subheader("Data quality flags")
                    for anomaly in result["anomalies"]:
                        st.warning(anomaly)

                    numeric_cols = df.select_dtypes(include='number').columns.tolist()
                    if len(numeric_cols) >= 1:
                        st.subheader("Quick chart")
                        st.bar_chart(df[numeric_cols[0]].head(20))

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Paste this error message to Claude for help fixing it!")
