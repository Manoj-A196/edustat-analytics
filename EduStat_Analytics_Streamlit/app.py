import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="EduStat Analytics", page_icon="📊", layout="wide")

st.markdown("""<style>
.main .block-container{padding-top:2rem;max-width:1400px}
[data-testid="stMetric"]{border:1px solid #e5e7eb;padding:15px;border-radius:12px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def sample_data():
    rows=[]
    traditional=[72,68,75,70,74,69,76,71,73,67,77,70,75,72,68]
    modern=[82,85,79,88,81,84,80,86,83,87,78,82,89,85,81]
    for i,score in enumerate(traditional+modern,1):
        method="Traditional" if i<=15 else "Modern"
        rows.append([f"S{i:03}",f"Student {i}",method,score,80+(i%17),2+(i%4),4])
    return pd.DataFrame(rows,columns=["Student_ID","Student_Name","Teaching_Method","Academic_Score","Attendance","Study_Hours","Semester"])

if "df" not in st.session_state: st.session_state.df=sample_data()
df=st.session_state.df

st.sidebar.title("📊 EduStat Analytics")
page=st.sidebar.radio("Navigation",["🏠 Dashboard","📂 Data Preprocessing","📈 t-Test Analysis","📊 Performance Visualization","📄 Reports","ℹ️ About Project"])
uploaded=st.sidebar.file_uploader("Upload CSV",type=["csv"])
if uploaded:
    try:
        st.session_state.df=pd.read_csv(uploaded); df=st.session_state.df
        st.sidebar.success("Dataset loaded.")
    except Exception as e: st.sidebar.error(str(e))
if st.sidebar.button("🔄 Load Sample Dataset"):
    st.session_state.df=sample_data(); st.rerun()

def get_analysis(data):
    cats=[c for c in data.columns if data[c].nunique(dropna=True)==2]
    nums=list(data.select_dtypes(include=np.number).columns)
    if not cats or not nums:return None
    gcol="Teaching_Method" if "Teaching_Method" in cats else cats[0]
    scol="Academic_Score" if "Academic_Score" in nums else nums[0]
    groups=list(data[gcol].dropna().unique())
    a=pd.to_numeric(data.loc[data[gcol]==groups[0],scol],errors="coerce").dropna()
    b=pd.to_numeric(data.loc[data[gcol]==groups[1],scol],errors="coerce").dropna()
    if len(a)<2 or len(b)<2:return None
    t,p=stats.ttest_ind(a,b,equal_var=True)
    return gcol,scol,groups,a,b,t,p

res=get_analysis(df)

if page=="🏠 Dashboard":
    st.title("EduStat Analytics")
    st.caption("Student Academic Performance & Statistical Analysis")
    st.header("Dashboard")
    if res:
        gcol,scol,groups,a,b,t,p=res
        c=st.columns(4)
        c[0].metric("Total Students",len(df)); c[1].metric("Group A",len(a)); c[2].metric("Group B",len(b)); c[3].metric("p-Value",f"{p:.4f}")
        c=st.columns(3)
        c[0].metric("Mean A",f"{a.mean():.2f}"); c[1].metric("Mean B",f"{b.mean():.2f}"); c[2].metric("t-Value",f"{t:.4f}")
        st.success("Statistically Significant Difference" if p<.05 else "No Statistically Significant Difference")
    st.info("Use the sidebar to upload data, preprocess it, perform the t-Test, visualize results, and generate a report.")

elif page=="📂 Data Preprocessing":
    st.header("Module 1 — Data Collection & Preprocessing")
    st.dataframe(df,use_container_width=True)
    c=st.columns(3); c[0].metric("Records",len(df)); c[1].metric("Missing Values",int(df.isna().sum().sum())); c[2].metric("Duplicates",int(df.duplicated().sum()))
    if st.button("🧹 Clean Data"):
        st.session_state.df=df.drop_duplicates().dropna(); st.success("Dataset cleaned."); st.rerun()
    cats=[c for c in df.columns if df[c].nunique(dropna=True)==2]
    nums=list(df.select_dtypes(include=np.number).columns)
    if cats and nums:
        gcol=st.selectbox("Grouping Variable",cats)
        scol=st.selectbox("Performance Variable",nums)
        groups=list(df[gcol].dropna().unique())
        if len(groups)>=2:
            a=pd.to_numeric(df.loc[df[gcol]==groups[0],scol],errors="coerce").dropna()
            b=pd.to_numeric(df.loc[df[gcol]==groups[1],scol],errors="coerce").dropna()
            st.subheader("Descriptive Statistics")
            out=pd.DataFrame({"Group":[groups[0],groups[1]],"Sample Size":[len(a),len(b)],"Mean":[a.mean(),b.mean()],"Std. Deviation":[a.std(),b.std()],"Minimum":[a.min(),b.min()],"Maximum":[a.max(),b.max()]})
            st.dataframe(out.round(3),use_container_width=True)

elif page=="📈 t-Test Analysis":
    st.header("Module 2 — Student's t-Test")
    st.markdown("**H₀:** There is no significant difference in mean academic performance between the two groups.")
    st.markdown("**H₁:** There is a significant difference in mean academic performance between the two groups.")
    confidence=st.selectbox("Confidence Level",[90,95,99],index=1)
    if res:
        gcol,scol,groups,a,b,t,p=res
        dof=len(a)+len(b)-2; diff=a.mean()-b.mean()
        se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)); alpha=1-confidence/100
        crit=stats.t.ppf(1-alpha/2,dof); lo,hi=diff-crit*se,diff+crit*se
        c=st.columns(4); c[0].metric("t-Value",f"{t:.4f}"); c[1].metric("p-Value",f"{p:.4f}"); c[2].metric("Degrees of Freedom",dof); c[3].metric("Mean Difference",f"{diff:.2f}")
        st.write(f"**{confidence}% Confidence Interval:** ({lo:.2f}, {hi:.2f})")
        if p<.05:
            st.success("Decision: Reject H₀ — statistically significant at α = 0.05.")
            st.write(f"The p-value ({p:.4f}) is less than 0.05, indicating evidence of a significant difference.")
        else:
            st.info("Decision: Do not reject H₀ — insufficient evidence of a significant difference.")
    else: st.warning("A valid two-group dataset is required.")

elif page=="📊 Performance Visualization":
    st.header("Module 3 — Performance Comparison & Result Visualization")
    if res:
        _,_,groups,a,b,_,_=res
        t1,t2,t3=st.tabs(["Bar Chart","Histogram","Box Plot"])
        with t1:
            fig,ax=plt.subplots(); ax.bar([str(groups[0]),str(groups[1])],[a.mean(),b.mean()]); ax.set_ylabel("Mean Academic Score"); st.pyplot(fig)
        with t2:
            fig,ax=plt.subplots(); ax.hist(a,alpha=.6,label=str(groups[0])); ax.hist(b,alpha=.6,label=str(groups[1])); ax.set_xlabel("Academic Score"); ax.legend(); st.pyplot(fig)
        with t3:
            fig,ax=plt.subplots(); ax.boxplot([a,b],labels=[str(groups[0]),str(groups[1])]); ax.set_ylabel("Academic Score"); st.pyplot(fig)
    else: st.warning("A valid two-group dataset is required.")

elif page=="📄 Reports":
    st.header("Statistical Analysis Report")
    if res:
        gcol,scol,groups,a,b,t,p=res
        decision="Reject H₀ — statistically significant" if p<.05 else "Do not reject H₀ — not statistically significant"
        report=f"""EDUSTAT ANALYTICS
Comparative Analysis of Student Academic Performance Under Different Teaching Methods Using Student's t-Test

Total Students: {len(df)}
Grouping Variable: {gcol}
Performance Variable: {scol}

Group A ({groups[0]}): n={len(a)}, mean={a.mean():.3f}, SD={a.std():.3f}
Group B ({groups[1]}): n={len(b)}, mean={b.mean():.3f}, SD={b.std():.3f}

t-value: {t:.5f}
p-value: {p:.5f}
Degrees of freedom: {len(a)+len(b)-2}
Decision: {decision}

Interpretation: {"The p-value is less than 0.05, so H₀ is rejected." if p<.05 else "The p-value is greater than or equal to 0.05, so H₀ is not rejected."}
"""
        st.text_area("Report Preview",report,height=420)
        st.download_button("⬇️ Download Report",report,"edustat_analysis_report.txt","text/plain")

else:
    st.header("About Project")
    st.subheader("Abstract")
    st.write("This project analyses student academic performance by comparing two independent groups using the Student's t-Test. The system preprocesses the student dataset, organizes the data into groups, calculates descriptive statistics, and presents hypothesis testing results through tables and visual charts.")
    st.subheader("Problem Statement")
    st.write("Student academic performance is influenced by multiple factors, making manual comparison time-consuming and potentially inaccurate. An efficient and user-friendly system is needed to analyze student performance using the Student's t-Test and provide data-driven insights.")
    st.subheader("Modules")
    st.write("**Module 1:** Student Academic Data Collection and Preprocessing")
    st.write("**Module 2:** Student's t-Test Statistical Analysis")
    st.write("**Module 3:** Performance Comparison and Result Visualization")
    st.divider()
    st.write("Submitted by: Pradeep G (192424252) · Manoj A (192421295) · Arun Kumar N (192424184)")
    st.write("Guided by: Dr. Balaji R")
    st.write("SIMATS Engineering · UBA5304")
