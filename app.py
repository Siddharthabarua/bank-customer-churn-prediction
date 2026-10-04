import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Bank Customer Churn Risk Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.stApp{background:#F4FAFF}
[data-testid="stSidebar"]{background:#E8F4FF}
.title{color:#145DA0;font-size:2.2rem;font-weight:800}
.subtitle{color:#47718F;margin-bottom:1.2rem}
.kpi{background:white;border:1px solid #CFE8FA;border-radius:15px;padding:16px;box-shadow:0 5px 18px #337dab12}
.kpi-label{color:#64849B;font-size:.85rem;font-weight:600}
.kpi-value{color:#145DA0;font-size:1.55rem;font-weight:800;margin-top:5px}
.low{background:#E8F8F0;color:#19734A;padding:14px;border-radius:12px;font-weight:700;text-align:center}
.medium{background:#FFF7DF;color:#946C00;padding:14px;border-radius:12px;font-weight:700;text-align:center}
.high{background:#FFECEC;color:#B33A3A;padding:14px;border-radius:12px;font-weight:700;text-align:center}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    for p in ("European_Bank.csv", "data/European_Bank.csv"):
        try:
            return pd.read_csv(p)
        except FileNotFoundError:
            pass
    raise FileNotFoundError("European_Bank.csv not found. Place it beside app.py or in data/.")

def engineer(d):
    d=d.copy()
    d["BalanceSalaryRatio"]=d["Balance"]/(d["EstimatedSalary"]+1)
    d["ProductDensity"]=d["NumOfProducts"]/(d["Tenure"]+1)
    d["EngagementProduct"]=d["IsActiveMember"]*d["NumOfProducts"]
    d["AgeTenure"]=d["Age"]*(d["Tenure"]+1)
    d["ZeroBalanceFlag"]=(d["Balance"]==0).astype(int)
    return d

FEATURES=["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts",
"HasCrCard","IsActiveMember","EstimatedSalary","BalanceSalaryRatio","ProductDensity",
"EngagementProduct","AgeTenure","ZeroBalanceFlag"]
CAT=["Geography","Gender"]
NUM=[x for x in FEATURES if x not in CAT]

@st.cache_resource
def train(df):
    d=engineer(df); X=d[FEATURES]; y=d["Exited"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("num",numeric,NUM),("cat",categorical,CAT)])
    pipe=Pipeline([("prep",prep),("model",GradientBoostingClassifier(
        n_estimators=250,learning_rate=.05,max_depth=3,random_state=42))])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    m={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),
       "Recall":recall_score(yte,pred,zero_division=0),"F1 Score":f1_score(yte,pred,zero_division=0),
       "ROC-AUC":roc_auc_score(yte,prob)}
    return pipe,m

def customer(credit,geo,gender,age,tenure,balance,products,card,active,salary):
    d=pd.DataFrame([{"CreditScore":credit,"Geography":geo,"Gender":gender,"Age":age,
       "Tenure":tenure,"Balance":balance,"NumOfProducts":products,"HasCrCard":card,
       "IsActiveMember":active,"EstimatedSalary":salary}])
    return engineer(d)[FEATURES]

def risk(p):
    return ("Low Risk","low") if p<.3 else (("Medium Risk","medium") if p<.6 else ("High Risk","high"))

try:
    df=load_data()
except Exception as e:
    st.error(str(e)); st.stop()

model,metrics=train(df)
st.markdown('<div class="title">🏦 Bank Customer Churn Risk Dashboard</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predictive modeling and risk scoring for early identification of customers likely to churn.</div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("Dashboard")
    st.caption("Gradient Boosting • Stratified 80/20 split")
    st.markdown("---")
    st.markdown("### Risk Bands")
    st.write("🟢 Low: probability < 30%")
    st.write("🟡 Medium: 30%–59.9%")
    st.write("🔴 High: ≥ 60%")
    st.markdown("---")
    st.write(f"Customers: **{len(df):,}**")
    st.write(f"Observed churn: **{df['Exited'].mean()*100:.2f}%**")

k1,k2,k3,k4=st.columns(4)
for col,label,value in [
    (k1,"Total Customers",f"{len(df):,}"),
    (k2,"Observed Churn",f"{df['Exited'].mean()*100:.1f}%"),
    (k3,"ROC-AUC",f"{metrics['ROC-AUC']:.3f}"),
    (k4,"Champion Model","Gradient Boosting")]:
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',unsafe_allow_html=True)

t1,t2,t3,t4=st.tabs(["🎯 Risk Calculator","📊 Portfolio Analytics","🔎 Model Explainability","🧪 What-if Simulator"])

with t1:
    st.subheader("Customer Risk Calculator")
    a,b,c=st.columns(3)
    with a:
        cr=st.number_input("Credit Score",300,850,650,key="calc_credit")
        age=st.number_input("Age",18,100,40,key="calc_age")
        ten=st.number_input("Tenure",0,10,5,key="calc_tenure")
    with b:
        geo=st.selectbox("Geography",sorted(df.Geography.dropna().unique()),key="calc_geo")
        gender=st.selectbox("Gender",sorted(df.Gender.dropna().unique()),key="calc_gender")
        products=st.slider("Number of Products",1,4,1,key="calc_products_slider")
    with c:
        bal=st.number_input("Balance",0.0,300000.0,75000.0,5000.0,key="calc_balance")
        sal=st.number_input("Estimated Salary",0.0,250000.0,100000.0,5000.0,key="calc_salary")
        card=st.selectbox("Has Credit Card?",["Yes","No"],key="calc_card")
        active=st.selectbox("Active Member?",["Yes","No"],key="calc_active")
    if st.button("Calculate Churn Risk",type="primary",use_container_width=True,key="calc_button"):
        p=float(model.predict_proba(customer(cr,geo,gender,age,ten,bal,products,
            int(card=="Yes"),int(active=="Yes"),sal))[0,1])
        band,cls=risk(p)
        st.markdown(f'<div class="{cls}">{band} — Estimated churn probability: {p*100:.1f}%</div>',unsafe_allow_html=True)
        st.progress(p)
        if p>=.6: st.warning("Prioritize this customer for retention outreach.")
        elif p>=.3: st.info("Monitor the customer and consider targeted engagement.")
        else: st.success("Standard relationship management is appropriate.")

with t2:
    st.subheader("Portfolio Analytics")
    left,right=st.columns(2)
    with left:
        counts=df["Exited"].value_counts().rename(index={0:"Retained",1:"Churned"}).reset_index()
        counts.columns=["Status","Customers"]
        st.plotly_chart(px.pie(counts,names="Status",values="Customers",hole=.5,title="Customer Churn Distribution"),use_container_width=True)
    with right:
        geo_data=df.groupby("Geography",as_index=False)["Exited"].mean().rename(columns={"Exited":"Churn Rate"})
        geo_data["Churn Rate"]*=100
        st.plotly_chart(px.bar(geo_data,x="Geography",y="Churn Rate",text_auto=".1f",title="Churn Rate by Geography",labels={"Churn Rate":"Churn Rate (%)"}),use_container_width=True)
    left,right=st.columns(2)
    with left:
        ad=df.groupby("IsActiveMember",as_index=False)["Exited"].mean()
        ad["Member Status"]=ad["IsActiveMember"].map({0:"Inactive",1:"Active"}); ad["Churn Rate"]=ad["Exited"]*100
        st.plotly_chart(px.bar(ad,x="Member Status",y="Churn Rate",text_auto=".1f",title="Churn Rate by Activity"),use_container_width=True)
    with right:
        pdx=df.groupby("NumOfProducts",as_index=False)["Exited"].mean(); pdx["Churn Rate"]=pdx["Exited"]*100
        st.plotly_chart(px.bar(pdx,x="NumOfProducts",y="Churn Rate",text_auto=".1f",title="Churn Rate by Number of Products"),use_container_width=True)

with t3:
    st.subheader("Model Explainability")
    prep=model.named_steps["prep"]; gb=model.named_steps["model"]
    imp=pd.DataFrame({"Feature":prep.get_feature_names_out(),"Importance":gb.feature_importances_})
    imp["Feature"]=imp["Feature"].str.replace("num__","",regex=False).str.replace("cat__","",regex=False)
    imp=imp.nlargest(12,"Importance").sort_values("Importance")
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Top Model Features"),use_container_width=True)
    perf=pd.DataFrame({"Metric":list(metrics.keys()),"Score":list(metrics.values())})
    st.plotly_chart(px.bar(perf,x="Metric",y="Score",text_auto=".3f",title="Gradient Boosting Performance").update_yaxes(range=[0,1]),use_container_width=True)
    st.info("Feature importance describes model behavior; it does not prove causation.")

with t4:
    st.subheader("🧪 What-if Customer Simulator")
    st.write("Change characteristics and observe the predicted churn risk.")
    x,y=st.columns(2)
    with x:
        wa=st.slider("Age",18,92,40,key="whatif_age")
        wt=st.slider("Tenure",0,10,5,key="whatif_tenure")
        wp=st.slider("Number of Products",1,4,1,key="whatif_products")
        wactive=st.selectbox("Active Member",["Yes","No"],key="whatif_active")
    with y:
        wb=st.slider("Balance",0,250000,75000,5000,key="whatif_balance")
        ws=st.slider("Estimated Salary",0,200000,100000,5000,key="whatif_salary")
        wc=st.slider("Credit Score",300,850,650,key="whatif_credit")
        wcard=st.selectbox("Has Credit Card",["Yes","No"],key="whatif_card")
    wg=st.selectbox("Geography",sorted(df.Geography.dropna().unique()),key="whatif_geo")
    wgender=st.selectbox("Gender",sorted(df.Gender.dropna().unique()),key="whatif_gender")
    p=float(model.predict_proba(customer(wc,wg,wgender,wa,wt,wb,wp,int(wcard=="Yes"),int(wactive=="Yes"),ws))[0,1])
    band,cls=risk(p)
    st.markdown(f'<div class="{cls}">{band} — Simulated churn probability: {p*100:.1f}%</div>',unsafe_allow_html=True)
    st.progress(p)
    if p>=.6: st.error("High-risk scenario: proactive retention is recommended.")
    elif p>=.3: st.warning("Medium-risk scenario: targeted engagement may be useful.")
    else: st.success("Low-risk scenario.")

st.markdown("---")
st.caption("Bank Customer Churn Prediction • Gradient Boosting • Educational/portfolio project.")
