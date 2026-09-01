import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="EduStat Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 2rem; max-width: 1400px;}
[data-testid="stMetric"] {border: 1px solid #e5e7eb; padding: 15px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def make_sample_data():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(1, 61):
        method = "Traditional" if i <= 30 else "Modern"
        score = int(np.clip(rng.normal(72 if method == "Traditional" else 80, 6), 45, 100))
        attendance = int(np.clip(rng.normal(84 if method == "Traditional" else 89, 5), 60, 100))
        study = round(float(np.clip(rng.normal(3 if method == "Traditional" else 4, 1), 1, 8)), 1)
        rows.append([f"S{i:03}", f"Student {i}", method, score, attendance, study, 4])
    return pd.DataFrame(rows, columns=[
        "Student_ID","Student_Name","Teaching_Method",
        "Academic_Score","Attendance","Study_Hours","Semester"
    ])

if "df" not in st.session_state:
    st.session_state.df = make_sample_data()

st.sidebar.title("📊 EduStat Analytics")
page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard", "📂 Data Preprocessing", "📈 t-Test Analysis",
    "📊 Performance Visualization", "📄 Reports", "ℹ️ About Project"
])

uploaded = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        st.session_state.df = new_df
        st.sidebar.success(f"New dataset loaded: {len(new_df)} rows")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

if st.sidebar.button("🔄 Load 60-Student Sample Dataset"):
    st.session_state.df = make_sample_data()
    st.rerun()

df = st.session_state.df

def get_analysis(data, gcol=None, scol=None):
    if gcol is None:
        cats = [c for c in data.columns if data[c].nunique(dropna=True) == 2]
        gcol = "Teaching_Method" if "Teaching_Method" in cats else (cats[0] if cats else None)
    if scol is None:
        nums = list(data.select_dtypes(include=np.number).columns)
        scol = "Academic_Score" if "Academic_Score" in nums else (nums[0] if nums else None)
    if not gcol or not scol:
        return None
    groups = list(data[gcol].dropna().unique())
    if len(groups) < 2:
        return None
    a = pd.to_numeric(data.loc[data[gcol] == groups[0], scol], errors="coerce").dropna()
    b = pd.to_numeric(data.loc[data[gcol] == groups[1], scol], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=True)
    return gcol, scol, groups[:2], a, b, t, p

res = get_analysis(df)

if page == "🏠 Dashboard":
    st.title("📊 EduStat Analytics")
    st.caption("Student Academic Performance & Statistical Analysis")
    st.header("Dashboard")
    if res:
        gcol, scol, groups, a, b, t, p = res
        c = st.columns(4)
        c[0].metric("Total Students", len(df))
        c[1].metric(f"Group A — {groups[0]}", len(a))
        c[2].metric(f"Group B — {groups[1]}", len(b))
        c[3].metric("p-Value", f"{p:.4f}")
        c = st.columns(3)
        c[0].metric("Mean A", f"{a.mean():.2f}")
        c[1].metric("Mean B", f"{b.mean():.2f}")
        c[2].metric("t-Value", f"{t:.4f}")
        if p < 0.05:
            st.success("Statistically Significant Difference — Reject H₀ at α = 0.05.")
        else:
            st.info("No Statistically Significant Difference — Do not reject H₀ at α = 0.05.")
    st.subheader("Current Dataset")
    st.dataframe(df, use_container_width=True)

elif page == "📂 Data Preprocessing":
    st.header("Module 1 — Student Data Collection & Preprocessing")
    st.write(f"**Current dataset:** {len(df)} records")
    st.dataframe(df, use_container_width=True)
    c = st.columns(3)
    c[0].metric("Records", len(df))
    c[1].metric("Missing Values", int(df.isna().sum().sum()))
    c[2].metric("Duplicate Rows", int(df.duplicated().sum()))

    if st.button("🧹 Clean Data"):
        cleaned = df.drop_duplicates().dropna().copy()
        st.session_state.df = cleaned
        st.success(f"Data cleaned. {len(df) - len(cleaned)} rows removed.")
        st.rerun()

    cats = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    nums = list(df.select_dtypes(include=np.number).columns)
    if cats and nums:
        gcol = st.selectbox("Grouping Variable", cats,
                            index=cats.index("Teaching_Method") if "Teaching_Method" in cats else 0)
        scol = st.selectbox("Performance Variable", nums,
                            index=nums.index("Academic_Score") if "Academic_Score" in nums else 0)
        groups = list(df[gcol].dropna().unique())
        if len(groups) >= 2:
            a = pd.to_numeric(df.loc[df[gcol] == groups[0], scol], errors="coerce").dropna()
            b = pd.to_numeric(df.loc[df[gcol] == groups[1], scol], errors="coerce").dropna()
            st.subheader("Descriptive Statistics")
            out = pd.DataFrame({
                "Group": [groups[0], groups[1]],
                "Sample Size": [len(a), len(b)],
                "Mean": [a.mean(), b.mean()],
                "Std. Deviation": [a.std(), b.std()],
                "Minimum": [a.min(), b.min()],
                "Maximum": [a.max(), b.max()]
            })
            st.dataframe(out.round(3), use_container_width=True)
    else:
        st.warning("Your CSV needs a categorical column with two groups and a numeric performance column.")

elif page == "📈 t-Test Analysis":
    st.header("Module 2 — Student's t-Test Statistical Analysis")
    st.markdown("**H₀:** There is no significant difference in mean academic performance between the two groups.")
    st.markdown("**H₁:** There is a significant difference in mean academic performance between the two groups.")
    confidence = st.selectbox("Confidence Level", [90, 95, 99], index=1)
    if res:
        gcol, scol, groups, a, b, t, p = res
        dof = len(a) + len(b) - 2
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        alpha = 1 - confidence/100
        critical = stats.t.ppf(1-alpha/2, dof)
        lo, hi = diff-critical*se, diff+critical*se
        c = st.columns(4)
        c[0].metric("t-Value", f"{t:.4f}")
        c[1].metric("p-Value", f"{p:.4f}")
        c[2].metric("Degrees of Freedom", dof)
        c[3].metric("Mean Difference", f"{diff:.2f}")
        st.write(f"**{confidence}% Confidence Interval:** ({lo:.2f}, {hi:.2f})")
        if p < 0.05:
            st.success("Decision: Reject H₀ — statistically significant at α = 0.05.")
            st.write(f"The p-value ({p:.4f}) is less than 0.05, indicating evidence of a significant difference.")
        else:
            st.info("Decision: Do not reject H₀ — insufficient evidence of a significant difference.")
    else:
        st.warning("A valid two-group dataset is required.")

elif page == "📊 Performance Visualization":
    st.header("Module 3 — Performance Comparison & Result Visualization")
    if res:
        _, _, groups, a, b, _, _ = res
        t1, t2, t3 = st.tabs(["Bar Chart", "Histogram", "Box Plot"])
        with t1:
            fig, ax = plt.subplots()
            ax.bar([str(groups[0]), str(groups[1])], [a.mean(), b.mean()])
            ax.set_ylabel("Mean Academic Score")
            ax.set_title("Mean Performance Comparison")
            st.pyplot(fig)
            plt.close(fig)
        with t2:
            fig, ax = plt.subplots()
            ax.hist(a, alpha=.6, label=str(groups[0]))
            ax.hist(b, alpha=.6, label=str(groups[1]))
            ax.set_xlabel("Academic Score")
            ax.set_ylabel("Students")
            ax.set_title("Academic Score Distribution")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        with t3:
            fig, ax = plt.subplots()
            # tick_labels fixes the Matplotlib deprecation/error shown on Streamlit Cloud
            ax.boxplot([a, b], tick_labels=[str(groups[0]), str(groups[1])])
            ax.set_ylabel("Academic Score")
            ax.set_title("Performance Distribution")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("A valid two-group dataset is required.")

elif page == "📄 Reports":
    st.header("📄 Statistical Analysis Report")
    if res:
        gcol, scol, groups, a, b, t, p = res
        decision = "Reject H₀ — statistically significant" if p < .05 else "Do not reject H₀ — not statistically significant"
        report = f"""EDUSTAT ANALYTICS
Student Academic Performance & Statistical Analysis

Total Students: {len(df)}
Grouping Variable: {gcol}
Performance Variable: {scol}

Group A ({groups[0]}): n={len(a)}, mean={a.mean():.3f}, SD={a.std():.3f}
Group B ({groups[1]}): n={len(b)}, mean={b.mean():.3f}, SD={b.std():.3f}

Independent Student's t-Test
t-value = {t:.5f}
p-value = {p:.5f}
Degrees of freedom = {len(a)+len(b)-2}

Decision: {decision}

Interpretation:
{"The p-value is less than 0.05, so H₀ is rejected." if p < .05 else "The p-value is greater than or equal to 0.05, so H₀ is not rejected."}
"""
        st.text_area("Report Preview", report, height=450)
        st.download_button("⬇️ Download Report", report, "edustat_analysis_report.txt", "text/plain")
    else:
        st.warning("A valid dataset is required.")

else:
    st.header("ℹ️ About Project")
    st.subheader("Abstract")
    st.write("This project analyses student academic performance by comparing two independent groups using the Student's t-Test. The system preprocesses the student dataset, organizes the data into groups, calculates descriptive statistics, and presents the results through tables, charts, and hypothesis-test outcomes.")
    st.subheader("Introduction")
    st.write("Student academic performance is an important indicator of learning effectiveness and educational quality. Factors such as teaching methods, study habits, attendance, and part-time employment can influence academic outcomes. Statistical techniques like the Student's t-Test help compare two independent groups.")
    st.subheader("Problem Statement")
    st.write("Student academic performance is influenced by multiple factors, making manual comparison time-consuming and potentially inaccurate. An efficient and user-friendly system is needed to analyze student performance using the Student's t-Test and provide data-driven insights.")
    st.subheader("Project Modules")
    st.write("**Module 1:** Student Academic Data Collection and Preprocessing")
    st.write("**Module 2:** Student's t-Test Statistical Analysis")
    st.write("**Module 3:** Performance Comparison and Result Visualization")
    st.divider()
    st.write("**Submitted by:** Pradeep G (192424252) · Manoj A (192421295) · Arun Kumar N (192424184)")
    st.write("**Guided by:** Dr. Balaji R")
    st.write("**SIMATS Engineering · UBA5304 – Probabilistic Methods and Linear Algebra**")
