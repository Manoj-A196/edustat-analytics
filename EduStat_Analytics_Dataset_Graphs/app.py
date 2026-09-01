import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="EduStat Analytics", page_icon="📊", layout="wide")

@st.cache_data
def sample_data():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(1, 61):
        method = "Traditional" if i <= 30 else "Modern"
        score = int(np.clip(rng.normal(72 if method == "Traditional" else 80, 6), 45, 100))
        attendance = int(np.clip(rng.normal(84 if method == "Traditional" else 89, 5), 60, 100))
        study = round(float(np.clip(rng.normal(3 if method == "Traditional" else 4.5, 1), 1, 8)), 1)
        rows.append([f"S{i:03}", f"Student {i}", method, score, attendance, study, 4])
    return pd.DataFrame(rows, columns=["Student_ID","Student_Name","Teaching_Method","Academic_Score","Attendance","Study_Hours","Semester"])

if "df" not in st.session_state:
    st.session_state.df = sample_data()

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

if st.sidebar.button("🔄 Load Sample Dataset"):
    st.session_state.df = sample_data()
    st.rerun()

df = st.session_state.df

def get_analysis(data, gcol=None, scol=None):
    cats = [c for c in data.columns if data[c].nunique(dropna=True) == 2]
    nums = list(data.select_dtypes(include=np.number).columns)
    if not cats or not nums: return None
    gcol = gcol or ("Teaching_Method" if "Teaching_Method" in cats else cats[0])
    scol = scol or ("Academic_Score" if "Academic_Score" in nums else nums[0])
    groups = list(data[gcol].dropna().unique())
    if len(groups) < 2: return None
    a = pd.to_numeric(data.loc[data[gcol] == groups[0], scol], errors="coerce").dropna()
    b = pd.to_numeric(data.loc[data[gcol] == groups[1], scol], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2: return None
    t, p = stats.ttest_ind(a, b, equal_var=True)
    return gcol, scol, groups[:2], a, b, t, p

res = get_analysis(df)

if page == "🏠 Dashboard":
    st.title("📊 EduStat Analytics")
    st.caption("Student Academic Performance & Statistical Analysis")
    if res:
        gcol, scol, groups, a, b, t, p = res
        c = st.columns(4)
        c[0].metric("Total Students", len(df))
        c[1].metric(f"{groups[0]}", len(a))
        c[2].metric(f"{groups[1]}", len(b))
        c[3].metric("p-Value", f"{p:.4f}")
        st.subheader("Current Uploaded Dataset")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Upload a valid two-group dataset.")

elif page == "📂 Data Preprocessing":
    st.header("Module 1 — Student Data Collection & Preprocessing")
    st.write(f"**Current dataset: {len(df)} records**")
    st.dataframe(df, use_container_width=True)
    c = st.columns(3)
    c[0].metric("Records", len(df))
    c[1].metric("Missing Values", int(df.isna().sum().sum()))
    c[2].metric("Duplicates", int(df.duplicated().sum()))
    if st.button("🧹 Clean Data"):
        st.session_state.df = df.drop_duplicates().dropna().copy()
        st.success("Data cleaned successfully.")
        st.rerun()

    cats = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    nums = list(df.select_dtypes(include=np.number).columns)
    if cats and nums:
        gcol = st.selectbox("Grouping Variable", cats, index=cats.index("Teaching_Method") if "Teaching_Method" in cats else 0)
        scol = st.selectbox("Performance Variable", nums, index=nums.index("Academic_Score") if "Academic_Score" in nums else 0)
        groups = list(df[gcol].dropna().unique())[:2]
        if len(groups) == 2:
            a = pd.to_numeric(df.loc[df[gcol] == groups[0], scol], errors="coerce").dropna()
            b = pd.to_numeric(df.loc[df[gcol] == groups[1], scol], errors="coerce").dropna()
            out = pd.DataFrame({"Group":[groups[0],groups[1]],"Sample Size":[len(a),len(b)],"Mean":[a.mean(),b.mean()],"Std. Deviation":[a.std(),b.std()]})
            st.subheader("Descriptive Statistics")
            st.dataframe(out.round(3), use_container_width=True)

elif page == "📈 t-Test Analysis":
    st.header("Module 2 — Student's t-Test Statistical Analysis")
    st.markdown("**H₀:** There is no significant difference in mean academic performance between the two groups.")
    st.markdown("**H₁:** There is a significant difference in mean academic performance between the two groups.")
    confidence = st.selectbox("Confidence Level", [90,95,99], index=1)
    if res:
        gcol, scol, groups, a, b, t, p = res
        dof = len(a)+len(b)-2
        diff = a.mean()-b.mean()
        se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        alpha = 1-confidence/100
        critical = stats.t.ppf(1-alpha/2,dof)
        lo, hi = diff-critical*se, diff+critical*se
        c=st.columns(4)
        c[0].metric("t-Value",f"{t:.4f}"); c[1].metric("p-Value",f"{p:.4f}"); c[2].metric("Degrees of Freedom",dof); c[3].metric("Mean Difference",f"{diff:.2f}")
        st.write(f"**{confidence}% Confidence Interval:** ({lo:.2f}, {hi:.2f})")
        if p < .05:
            st.success("Decision: Reject H₀ — statistically significant at α = 0.05.")
        else:
            st.info("Decision: Do not reject H₀ — insufficient evidence of a significant difference.")

elif page == "📊 Performance Visualization":
    st.header("Module 3 — Performance Comparison & Result Visualization")
    if res:
        gcol, scol, groups, a, b, t, p = res

        # Show which dataset is being visualized
        st.success(f"Graphs are generated from the CURRENT dataset: {len(df)} rows")
        st.caption(f"Grouping column: {gcol}  |  Performance column: {scol}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Individual Student Scores",
            "📈 Mean Comparison",
            "📉 Histogram",
            "🔲 Box Plot"
        ])

        with tab1:
            # Actual uploaded dataset: every student's score is plotted.
            plot_df = df[[gcol, scol]].copy()
            plot_df[scol] = pd.to_numeric(plot_df[scol], errors="coerce")
            plot_df = plot_df.dropna().reset_index(drop=True)
            plot_df["Student"] = range(1, len(plot_df)+1)
            fig, ax = plt.subplots(figsize=(11,5))
            for group in plot_df[gcol].dropna().unique()[:2]:
                part = plot_df[plot_df[gcol] == group]
                ax.scatter(part["Student"], part[scol], label=str(group), s=45)
            ax.set_xlabel("Student Record Number")
            ax.set_ylabel(scol)
            ax.set_title("Academic Score of Every Student in Uploaded Dataset")
            ax.legend()
            ax.grid(alpha=.2)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Actual Dataset Values")
            st.dataframe(df, use_container_width=True)

        with tab2:
            fig, ax = plt.subplots(figsize=(8,5))
            ax.bar([str(groups[0]), str(groups[1])], [a.mean(), b.mean()])
            ax.set_ylabel(f"Mean {scol}")
            ax.set_title("Mean Performance Comparison")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(9,5))
            ax.hist(a, bins=8, alpha=.65, label=str(groups[0]))
            ax.hist(b, bins=8, alpha=.65, label=str(groups[1]))
            ax.set_xlabel(scol)
            ax.set_ylabel("Number of Students")
            ax.set_title("Score Distribution from Current Dataset")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            fig, ax = plt.subplots(figsize=(8,5))
            ax.boxplot([a,b], tick_labels=[str(groups[0]),str(groups[1])])
            ax.set_ylabel(scol)
            ax.set_title("Performance Distribution from Current Dataset")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("A valid dataset with two groups and a numeric performance column is required.")

elif page == "📄 Reports":
    st.header("📄 Statistical Analysis Report")
    if res:
        gcol,scol,groups,a,b,t,p=res
        report=f"""EDUSTAT ANALYTICS
Comparative Analysis of Student Academic Performance

Dataset: {len(df)} records
Grouping Variable: {gcol}
Performance Variable: {scol}

Group A ({groups[0]}): n={len(a)}, mean={a.mean():.3f}, SD={a.std():.3f}
Group B ({groups[1]}): n={len(b)}, mean={b.mean():.3f}, SD={b.std():.3f}

Independent Student's t-Test
t-value = {t:.5f}
p-value = {p:.5f}
Degrees of freedom = {len(a)+len(b)-2}

Decision: {"Reject H₀ — statistically significant" if p<.05 else "Do not reject H₀ — not statistically significant"}
"""
        st.text_area("Report Preview",report,height=400)
        st.download_button("⬇️ Download Report",report,"edustat_analysis_report.txt","text/plain")

else:
    st.header("ℹ️ About Project")
    st.subheader("Abstract")
    st.write("This project analyses student academic performance by comparing two independent groups using the Student's t-Test. The system preprocesses student data, calculates descriptive statistics, performs the t-Test, and presents results through tables and visual charts.")
    st.subheader("Project Modules")
    st.write("**Module 1:** Student Academic Data Collection and Preprocessing")
    st.write("**Module 2:** Student's t-Test Statistical Analysis")
    st.write("**Module 3:** Performance Comparison and Result Visualization")
    st.divider()
    st.write("Submitted by: Pradeep G (192424252) · Manoj A (192421295) · Arun Kumar N (192424184)")
    st.write("Guided by: Dr. Balaji R")
