import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Global Health Analytics Dashboard", layout="wide")

# 2. Optimized Data Loading (Matching Notebook Logic)
@st.cache_data
def load_data():
    df = pd.read_csv('Global Health Statistics.csv')
    df.columns = df.columns.str.strip()
    
    # Fill missing values with median (as per Notebook Cell 5)
    df = df.fillna(df.median(numeric_only=True))
    
    # Ensure numeric types for analysis
    num_cols = ['Mortality Rate (%)', 'Recovery Rate (%)', 'Average Treatment Cost (USD)', 
                'Population Affected', 'Prevalence Rate (%)', 'Healthcare Access (%)', 
                'Doctors per 1000', 'Hospital Beds per 1000']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Map 'Region' if missing (Safety check)
    if 'Region' not in df.columns and 'Country' in df.columns:
        df['Region'] = df['Country']
        
    # Categorize Access for Lollipop Chart (as per Notebook Cell 29)
    if 'Healthcare Access (%)' in df.columns:
        df['Access_Level'] = pd.qcut(df['Healthcare Access (%)'], 3, labels=['Low', 'Medium', 'High'])
        
    return df

df = load_data()

# 3. Sidebar Navigation (Clean Titles, No "Q1/Q2")
st.sidebar.title("Healthcare Analytics")
st.sidebar.info("Select a specific analysis from the project below.")

menu_options = [
    "Mortality Extremes by Country",        # Q1
    "Disease Prevalence Over Time",         # Q2
    "Treatment Costs by Category",          # Q3
    "Healthcare Efficiency Score",          # Q4
    "Infrastructure vs Recovery Rate",      # Q5
    "Cost Volatility Analysis",             # Q6
    "Top 10 Most Common Diseases",          # Q7
    "Gender Recovery Gap Analysis",         # Q8
    "Global Resource Comparison",           # Q9
    "Access Level vs Recovery Impact",      # Q10
    "Cost vs Success (Bubble Analysis)"     # Q11
]

choice = st.sidebar.radio("Navigation Menu", menu_options)

st.title(choice)

# --- EXACT NOTEBOOK VISUALIZATION LOGIC ---

if choice == "Mortality Extremes by Country":
    country_avg = df.groupby('Country')['Mortality Rate (%)'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.bar(range(len(country_avg)), country_avg.values, color='steelblue')
    plt.xticks(range(len(country_avg)), country_avg.index, rotation=90)
    plt.ylim(country_avg.min() - 0.1, country_avg.max() + 0.1)
    plt.ylabel('Mortality Rate (%)')
    st.pyplot(fig)

elif choice == "Disease Prevalence Over Time":
    yearly_trend = df.groupby('Year')['Prevalence Rate (%)'].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(yearly_trend.index, yearly_trend.values, marker='o', color='darkred', linewidth=2)
    plt.grid(alpha=0.3)
    plt.ylabel("Prevalence Rate (%)")
    st.pyplot(fig)

elif choice == "Treatment Costs by Category":
    cost_data = df.groupby('Disease Category')['Average Treatment Cost (USD)'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = plt.bar(cost_data.index, cost_data.values, color='skyblue')
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():.0f}', ha='center', va='bottom')
    st.pyplot(fig)

elif choice == "Healthcare Efficiency Score":
    df['Efficiency_Score'] = df['Recovery Rate (%)'] / df['Mortality Rate (%)']
    top_10 = df.groupby('Country')['Efficiency_Score'].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(top_10.index, top_10.values, color='dodgerblue')
    plt.ylabel("Efficiency (Recovery/Mortality)")
    st.pyplot(fig)

elif choice == "Infrastructure vs Recovery Rate":
    group_col = 'Region' if 'Region' in df.columns else 'Country'
    infra = df.groupby(group_col)[['Hospital Beds per 1000', 'Recovery Rate (%)']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=infra, x='Hospital Beds per 1000', y='Recovery Rate (%)', hue=group_col, s=200)
    st.pyplot(fig)

elif choice == "Cost Volatility Analysis":
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df.sample(2000), x='Disease Category', y='Average Treatment Cost (USD)', hue='Disease Category', palette='Set2', legend=False)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif choice == "Top 10 Most Common Diseases":
    top_diseases = df['Disease Name'].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_diseases.plot(kind='barh', color='steelblue', alpha=0.8)
    for i, count in enumerate(top_diseases):
        plt.text(count + 5, i, str(count), va='center')
    st.pyplot(fig)

elif choice == "Gender Recovery Gap Analysis":
    df_sample = df.sample(n=min(50000, len(df)))
    gap_data = df_sample.groupby(['Disease Category', 'Gender'])['Recovery Rate (%)'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=gap_data, x='Disease Category', y='Recovery Rate (%)', hue='Gender')
    plt.ylim(gap_data['Recovery Rate (%)'].min() * 0.98, gap_data['Recovery Rate (%)'].max() * 1.02)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif choice == "Global Resource Comparison":
    countries = ['USA', 'Japan', 'Nigeria', 'Germany', 'India']
    m_df = df[df['Country'].isin(countries)].groupby('Country')[['Healthcare Access (%)', 'Doctors per 1000', 'Hospital Beds per 1000']].mean().reindex(countries).reset_index()
    x = np.arange(len(countries))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, m_df['Healthcare Access (%)'], width, label='Access %', color='blue')
    ax.bar(x, m_df['Doctors per 1000'], width, label='Doctors', color='green')
    ax.bar(x + width, m_df['Hospital Beds per 1000'], width, label='Beds', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend()
    st.pyplot(fig)

elif choice == "Access Level vs Recovery Impact":
    summary = df.groupby('Access_Level', observed=False)['Recovery Rate (%)'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.vlines(x=summary['Access_Level'], ymin=0, ymax=summary['Recovery Rate (%)'], color='skyblue', linewidth=2)
    plt.scatter(summary['Access_Level'], summary['Recovery Rate (%)'], color='navy', s=150, zorder=3)
    plt.ylim(summary['Recovery Rate (%)'].min() - 2, summary['Recovery Rate (%)'].max() + 2)
    st.pyplot(fig)

elif choice == "Cost vs Success (Bubble Analysis)":
    summary = df.groupby('Disease Category').agg({
        'Average Treatment Cost (USD)': 'mean', 'Recovery Rate (%)': 'mean', 'Population Affected': 'sum'
    }).reset_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    size = summary['Population Affected'] / summary['Population Affected'].max() * 2000 
    plt.scatter(summary.iloc[:,1], summary.iloc[:,2], s=size, alpha=0.6, c=range(len(summary)), cmap='viridis', edgecolors='black')
    for i, txt in enumerate(summary['Disease Category']):
        plt.annotate(txt, (summary.iloc[i,1], summary.iloc[i,2]), fontweight='bold')
    plt.axhline(summary.iloc[:,2].mean(), color='red', linestyle='--', alpha=0.5)
    plt.axvline(summary.iloc[:,1].mean(), color='red', linestyle='--', alpha=0.5)
    st.pyplot(fig)