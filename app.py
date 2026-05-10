import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SaaS Funnel Dashboard",
    layout="wide"
)

st.title("SaaS User Behavior Funnel Analysis Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(
    "cleaned_saas_user_behavior_dataset.csv"
)

# Convert datetime
df['event_timestamp'] = pd.to_datetime(
    df['event_timestamp']
)

df['signup_month'] = pd.to_datetime(
    df['signup_month']
)

# -----------------------------
# SIDEBAR FILTER
# -----------------------------
st.sidebar.header("Filters")

channel_filter = st.sidebar.multiselect(
    "Select Acquisition Channel",
    options=df['acquisition_channel'].unique(),
    default=df['acquisition_channel'].unique()
)

filtered_df = df[
    df['acquisition_channel']
    .isin(channel_filter)
]

# -----------------------------
# FUNNEL ANALYSIS
# -----------------------------
st.header("Conversion Funnel")

funnel_steps = [
    'Landing',
    'Sign-up',
    'Onboarding',
    'First Feature Use',
    'Upgrade'
]

funnel_counts = []

for step in funnel_steps:
    users = filtered_df[
        filtered_df['event_name'] == step
    ]['user_id'].nunique()

    funnel_counts.append(users)

funnel_df = pd.DataFrame({
    'Step': funnel_steps,
    'Users': funnel_counts
})

# Conversion Rate
conversion_rates = [100]

for i in range(1, len(funnel_counts)):
    rate = (
        funnel_counts[i]
        / funnel_counts[i - 1]
    ) * 100

    conversion_rates.append(rate)

funnel_df[
    'Conversion Rate (%)'
] = conversion_rates

# Drop-Off Rate
drop_rates = [0]

for i in range(1, len(funnel_counts)):
    drop = 100 - conversion_rates[i]
    drop_rates.append(drop)

funnel_df[
    'Drop-Off Rate (%)'
] = drop_rates

# Show table
st.dataframe(funnel_df)

# Funnel Chart
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    x='Step',
    y='Users',
    data=funnel_df,
    ax=ax
)

plt.xticks(rotation=20)
plt.title("User Funnel")

st.pyplot(fig)

# -----------------------------
# COHORT ANALYSIS
# -----------------------------
st.header("Cohort Analysis")

cohort_users = filtered_df.groupby(
    filtered_df['signup_month']
    .dt.to_period('M')
)['user_id'].nunique()

upgrade_users = filtered_df[
    filtered_df['event_name']
    == 'Upgrade'
]

cohort_upgrade = upgrade_users.groupby(
    upgrade_users['signup_month']
    .dt.to_period('M')
)['user_id'].nunique()

cohort_table = pd.DataFrame({
    'Total Users': cohort_users,
    'Upgraded Users': cohort_upgrade
}).fillna(0)

cohort_table[
    'Upgrade Rate (%)'
] = (
    cohort_table['Upgraded Users']
    / cohort_table['Total Users']
) * 100

st.dataframe(cohort_table)

fig, ax = plt.subplots(figsize=(8,5))

cohort_table[
    'Upgrade Rate (%)'
].plot(
    marker='o',
    ax=ax
)

plt.title(
    'Monthly Cohort Upgrade Rate'
)

st.pyplot(fig)

# -----------------------------
# SEGMENTATION ANALYSIS
# -----------------------------
st.header("Segmentation Analysis")

channel_users = filtered_df.groupby(
    'acquisition_channel'
)['user_id'].nunique()

upgrade_channel = filtered_df[
    filtered_df['event_name']
    == 'Upgrade'
]

channel_upgrade = upgrade_channel.groupby(
    'acquisition_channel'
)['user_id'].nunique()

segment_table = pd.DataFrame({
    'Total Users': channel_users,
    'Upgraded Users': channel_upgrade
}).fillna(0)

segment_table[
    'Upgrade Rate (%)'
] = (
    segment_table['Upgraded Users']
    / segment_table['Total Users']
) * 100

st.dataframe(segment_table)

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    x=segment_table.index,
    y=segment_table[
        'Upgrade Rate (%)'
    ],
    ax=ax
)

plt.title(
    'Upgrade Rate by Channel'
)

st.pyplot(fig)

# ==================================================
# CHURN PREDICTION DASHBOARD
# ==================================================

st.header("Churn Prediction Dashboard")

# --------------------------------
# USER LEVEL DATA
# --------------------------------

user_df = filtered_df.groupby(
    'user_id'
).agg({
    'acquisition_channel': 'first',
    'device_type': 'first',
    'country': 'first',
    'trial_status': 'first'
}).reset_index()

# Create target variable
upgrade_users = filtered_df[
    filtered_df['event_name']
    == 'Upgrade'
]['user_id'].unique()

# Churn column
user_df['churn'] = (
    ~user_df['user_id']
    .isin(upgrade_users)
).astype(int)

# --------------------------------
# LABEL ENCODING
# --------------------------------

encoder = LabelEncoder()

categorical_cols = [
    'acquisition_channel',
    'device_type',
    'country',
    'trial_status'
]

for col in categorical_cols:
    user_df[col] = encoder.fit_transform(
        user_df[col]
    )

# --------------------------------
# FEATURES & TARGET
# --------------------------------

X = user_df.drop(
    ['user_id', 'churn'],
    axis=1
)

y = user_df['churn']

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# --------------------------------
# TRAIN MODEL
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.subheader("Model Accuracy")

st.metric(
    label="Accuracy",
    value=f"{accuracy*100:.2f}%"
)

# --------------------------------
# CHURN DISTRIBUTION
# --------------------------------

st.subheader("Churn Distribution")

churn_counts = user_df[
    'churn'
].value_counts()

fig, ax = plt.subplots(
    figsize=(6,4)
)

sns.barplot(
    x=churn_counts.index,
    y=churn_counts.values,
    ax=ax
)

ax.set_xticklabels([
    'Retained',
    'Churned'
])

ax.set_ylabel("Users")

plt.title(
    'Churn Distribution'
)

st.pyplot(fig)

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------

st.subheader(
    "Feature Importance"
)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance':
    model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

fig, ax = plt.subplots(
    figsize=(8,5)
)

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance,
    ax=ax
)

plt.title(
    'Factors Affecting Churn'
)

st.pyplot(fig)

# --------------------------------
# USER CHURN PREDICTION
# --------------------------------

st.subheader(
    "Predict User Churn"
)

channel = st.selectbox(
    "Acquisition Channel",
    ['Organic',
     'Paid',
     'Referral']
)

device = st.selectbox(
    "Device Type",
    ['Desktop',
     'Mobile',
     'Tablet']
)

country = st.selectbox(
    "Country",
    ['Pakistan',
     'India',
     'UAE',
     'UK',
     'USA',
     'Unknown']
)

trial = st.selectbox(
    "Trial Status",
    ['Active',
     'Expired']
)

if st.button(
    "Predict Churn"
):

    input_data = pd.DataFrame({
        'acquisition_channel':
        [channel],
        'device_type':
        [device],
        'country':
        [country],
        'trial_status':
        [trial]
    })

    # Encode input
    for col in input_data.columns:
        input_data[col] = encoder.fit_transform(
            input_data[col]
        )

    prediction = model.predict(
        input_data
    )[0]

    if prediction == 1:
        st.error(
            "High chance of churn!"
        )
    else:
        st.success(
            "User likely to retain."
        )

# -----------------------------
# FINAL INSIGHTS
# -----------------------------
st.header("Key Insights")

highest_drop = funnel_df.loc[
    funnel_df[
        'Drop-Off Rate (%)'
    ].idxmax()
]

st.write(
    f"Highest drop-off occurs at **{highest_drop['Step']}** "
    f"with **{highest_drop['Drop-Off Rate (%)']:.2f}%** drop-off."
)

best_channel = segment_table[
    'Upgrade Rate (%)'
].idxmax()

st.write(
    f"Best acquisition channel is "
    f"**{best_channel}**."
)