import re
import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

warnings.filterwarnings('ignore')

st.set_page_config(page_title="GCP FinOps", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding: 0rem 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .divider {
        border-top: 1px solid #e5e5e5;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1.25rem;
        background: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #666;
        padding: 0.5rem 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
    }
    
    .stButton button {
        background: white;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        color: #1a1a1a;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        border-color: #1a1a1a;
        background: #fafafa;
    }
    
    .stButton button[kind="primary"] {
        background: #1a1a1a;
        border-color: #1a1a1a;
        color: white;
    }
    
    .stButton button[kind="primary"]:hover {
        background: #333;
    }
    
    .dataframe {
        font-size: 0.85rem;
    }
    
    .stAlert {
        background: #f8f8f8;
        border-left: 3px solid #999;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem;'>GCP FinOps</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; margin-bottom: 2rem;'>Hybrid anomaly detection for cloud cost optimization</p>", unsafe_allow_html=True)

for key, default in [
    ("engine", None),
    ("results_df", None),
    ("contamination", 0.05),
    ("chatops_history", []),
    ("chatops_prompt", ""),
    ("simulation_run", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

class IntelligentFinOpsEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.version = "3.0"

    def advanced_preprocess(self, df):
        df = df.copy()

        for col in ['Usage Start Date', 'Usage End Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df = df.dropna(subset=['Usage Start Date'])
        df['start_date'] = df['Usage Start Date'].dt.date
        df['month'] = df['Usage Start Date'].dt.strftime('%Y-%m')
        df['week'] = df['Usage Start Date'].dt.isocalendar().week

        if 'Usage End Date' in df.columns:
            df['duration_hours'] = (df['Usage End Date'] - df['Usage Start Date']).dt.total_seconds() / 3600
            df['duration_hours'] = df['duration_hours'].clip(lower=0.001)
        else:
            df['duration_hours'] = 1.0

        num_cols = ['Unrounded Cost ($)', 'Usage Quantity', 'CPU Utilization (%)',
                    'Memory Utilization (%)', 'Network Inbound Data (Bytes)', 'Network Outbound Data (Bytes)']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['utilization_score'] = (df.get('CPU Utilization (%)', 0) + df.get('Memory Utilization (%)', 0)) / 2
        df['cost_per_hour'] = df['Unrounded Cost ($)'] / df['duration_hours']
        df['total_network_gb'] = (df.get('Network Inbound Data (Bytes)', 0) +
                                  df.get('Network Outbound Data (Bytes)', 0)) / 1e9
        df['network_intensity'] = df['total_network_gb'] / df['duration_hours']

        if 'start_date' in df.columns:
            daily = df.groupby('start_date').agg({
                'Unrounded Cost ($)': 'sum',
                'Usage Quantity': 'sum'
            }).reset_index()
            daily['rolling_mean_cost'] = daily['Unrounded Cost ($)'].rolling(7, min_periods=3).mean()
            daily['rolling_mean_usage'] = daily['Usage Quantity'].rolling(7, min_periods=3).mean()
            df = df.merge(daily[['start_date', 'rolling_mean_cost', 'rolling_mean_usage']], on='start_date', how='left')
            df['cost_spike_ratio'] = df['Unrounded Cost ($)'] / (df['rolling_mean_cost'] + 0.001)
            df['usage_spike_ratio'] = df['Usage Quantity'] / (df['rolling_mean_usage'] + 0.001)

        df['usage_efficiency_score'] = df['utilization_score'] / (df['cost_per_hour'] + 0.001)

        for grp in ['Service Name', 'Region/Zone']:
            if grp in df.columns:
                stats = df.groupby(grp)['Unrounded Cost ($)'].agg(['mean', 'std']).fillna(0)
                stats.columns = [f'{grp.lower()}_avg_cost', f'{grp.lower()}_std_cost']
                df = df.merge(stats, on=grp, how='left')
                df[f'cost_vs_{grp.lower()}_avg'] = df['Unrounded Cost ($)'] / (df[f'{grp.lower()}_avg_cost'] + 0.001)

        return df.fillna(0)

    def detect_anomalies_ml(self, df):
        df = self.advanced_preprocess(df)

        base_features = ['Unrounded Cost ($)', 'utilization_score', 'network_intensity',
                         'cost_per_hour', 'cost_spike_ratio', 'usage_spike_ratio',
                         'cost_vs_service name_avg', 'cost_vs_region/zone_avg']
        features = [f for f in base_features if f in df.columns]

        X = df[features].values
        X = np.nan_to_num(X, nan=0.0)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = IsolationForest(contamination=st.session_state.contamination,
                                     random_state=42, n_estimators=300)
        self.model.fit(X_scaled)

        df['anomaly_score'] = self.model.score_samples(X_scaled)
        df['is_anomaly'] = self.model.predict(X_scaled)

        return df

    def add_rule_based_anomalies(self, df):
        daily_cost = df.groupby('start_date')['Unrounded Cost ($)'].sum().reset_index()
        daily_cost['rolling_mean_cost'] = daily_cost['Unrounded Cost ($)'].rolling(7, min_periods=1).mean()
        daily_cost['rolling_std_cost'] = daily_cost['Unrounded Cost ($)'].rolling(7, min_periods=1).std()
        daily_cost['daily_spike_flag'] = ((daily_cost['Unrounded Cost ($)'] > daily_cost['rolling_mean_cost'] * 1.5) & 
                                           (daily_cost['Unrounded Cost ($)'] > daily_cost['rolling_mean_cost'] + daily_cost['rolling_std_cost'])).astype(int)
        df = df.merge(daily_cost[['start_date', 'daily_spike_flag']], on='start_date', how='left')
        df['daily_spike_flag'] = df['daily_spike_flag'].fillna(0).astype(int)

        if 'Service Name' in df.columns:
            service_avg = df.groupby('Service Name')['Unrounded Cost ($)'].transform('mean')
            service_std = df.groupby('Service Name')['Unrounded Cost ($)'].transform('std')
            df['service_deviation_flag'] = ((df['Unrounded Cost ($)'] > service_avg * 2) & 
                                             (df['Unrounded Cost ($)'] > service_avg + 2 * service_std)).astype(int)
        else:
            df['service_deviation_flag'] = 0

        if 'Region/Zone' in df.columns:
            region_avg = df.groupby('Region/Zone')['Unrounded Cost ($)'].transform('mean')
            region_std = df.groupby('Region/Zone')['Unrounded Cost ($)'].transform('std')
            df['region_spike_flag'] = ((df['Unrounded Cost ($)'] > region_avg * 2) & 
                                        (df['Unrounded Cost ($)'] > region_avg + 2 * region_std)).astype(int)
        else:
            df['region_spike_flag'] = 0

        cpu_col = df['CPU Utilization (%)'] if 'CPU Utilization (%)' in df.columns else pd.Series(0, index=df.index)
        df['underutilized_flag'] = ((cpu_col < 30) & (df['Unrounded Cost ($)'] > 50)).astype(int)

        mean_usage = df['Usage Quantity'].mean() if 'Usage Quantity' in df.columns else 1
        std_usage = df['Usage Quantity'].std() if 'Usage Quantity' in df.columns else 1
        df['usage_surge_flag'] = (df['Usage Quantity'] > mean_usage + 2 * std_usage).astype(int)

        df['rule_based_score'] = (df['daily_spike_flag'] + df['service_deviation_flag'] +
                                  df['region_spike_flag'] + df['underutilized_flag'] +
                                  df['usage_surge_flag'])

        df['final_anomaly_flag'] = ((df['is_anomaly'] == -1) | (df['rule_based_score'] >= 1)).astype(int)
        
        df['severity'] = 'Low'
        df.loc[df['rule_based_score'] >= 3, 'severity'] = 'Critical'
        df.loc[df['rule_based_score'] == 2, 'severity'] = 'High'
        df.loc[df['rule_based_score'] == 1, 'severity'] = 'Medium'
        
        df['root_cause'] = df.apply(self.generate_root_cause, axis=1)
        df['recommendation'] = df.apply(self.generate_recommendation, axis=1)
        df['est_savings'] = (df['Unrounded Cost ($)'] * 0.35).round(2)

        return df

    def generate_root_cause(self, row):
        causes = []
        if row.get('daily_spike_flag', 0) == 1:       causes.append("Daily Cost Spike")
        if row.get('service_deviation_flag', 0) == 1:  causes.append("Service-Level Deviation")
        if row.get('region_spike_flag', 0) == 1:       causes.append("Region Cost Spike")
        if row.get('underutilized_flag', 0) == 1:      causes.append("Underutilized Resource")
        if row.get('usage_surge_flag', 0) == 1:        causes.append("Usage Surge")
        return " | ".join(causes) if causes else "ML-detected pattern anomaly"

    def generate_recommendation(self, row):
        if row.get('underutilized_flag', 0) == 1:
            return "Rightsize or terminate idle resources"
        elif row.get('daily_spike_flag', 0) == 1:
            return "Review deployments and auto-scaling policies"
        elif row.get('service_deviation_flag', 0) == 1:
            return "Optimize service configuration"
        elif row.get('region_spike_flag', 0) == 1:
            return "Consider multi-region optimization"
        elif row.get('usage_surge_flag', 0) == 1:
            return "Analyze traffic patterns and implement auto-scaling"
        return "Review resource usage patterns"

    def predict(self, df):
        df = self.detect_anomalies_ml(df)
        df = self.add_rule_based_anomalies(df)
        return df

def render_chatops():
    st.markdown("### ChatOps")
    st.markdown("<p style='color: #666; margin-bottom: 1rem;'>Natural language queries about your cost data</p>", unsafe_allow_html=True)
    
    for entry in st.session_state["chatops_history"]:
        with st.chat_message("user"):
            st.markdown(f"<p style='color: #666; font-size: 0.75rem;'>{entry['time']}</p>{entry['user']}", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            response = entry["response"]
            st.markdown(response["message"])
            
            if response.get("data") is not None and isinstance(response["data"], pd.DataFrame) and not response["data"].empty:
                st.dataframe(response["data"], use_container_width=True, hide_index=True)
            
            if response.get("chart") is not None:
                st.plotly_chart(response["chart"], use_container_width=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input(
            "Query",
            key="chat_input",
            placeholder="e.g., Show anomalies, Total cost, Top services",
            label_visibility="collapsed"
        )
    with col2:
        send = st.button("Send", type="primary", use_container_width=True)
    
    if send and prompt:
        response = _process_chatops_query(prompt.strip(), st.session_state.results_df)
        st.session_state["chatops_history"].append({
            "user": prompt,
            "response": response,
            "time": datetime.now().strftime("%H:%M")
        })
        st.rerun()

def _process_chatops_query(prompt: str, df: Optional[pd.DataFrame]) -> dict:
    if df is None:
        return {"message": "No dataset loaded. Upload a CSV file in the Overview tab."}
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['anomal', 'spike', 'flag']):
        return _get_anomalies_response(df)
    elif any(word in prompt_lower for word in ['total', 'summary', 'overview', 'cost']):
        return _get_summary_response(df)
    elif any(word in prompt_lower for word in ['top', 'expensive', 'highest']):
        return _get_top_services_response(df)
    elif any(word in prompt_lower for word in ['recommend', 'optimiz', 'save', 'saving']):
        return _get_recommendations_response(df)
    elif any(word in prompt_lower for word in ['region', 'location']):
        return _get_region_response(df)
    elif any(word in prompt_lower for word in ['trend', 'time', 'daily']):
        return _get_trend_response(df)
    else:
        return {"message": "Available queries: anomalies, cost summary, top services, recommendations, region analysis, trends"}

def _get_anomalies_response(df):
    if 'final_anomaly_flag' not in df.columns:
        return {"message": "Run anomaly detection by uploading a dataset in the Overview tab."}
    
    anomalies = df[df['final_anomaly_flag'] == 1]
    if anomalies.empty:
        return {"message": "No anomalies detected."}
    
    severity_counts = anomalies['severity'].value_counts()
    severity_text = " | ".join([f"{k}: {v}" for k, v in severity_counts.items()])
    
    display_cols = ['severity', 'root_cause', 'Service Name', 'Unrounded Cost ($)', 'recommendation']
    available_cols = [c for c in display_cols if c in anomalies.columns]
    
    return {
        "message": f"**{len(anomalies)} anomalies detected**\n\nSeverity: {severity_text}",
        "data": anomalies[available_cols].sort_values('Unrounded Cost ($)', ascending=False).head(10)
    }

def _get_summary_response(df):
    total = df['Unrounded Cost ($)'].sum()
    avg = df['Unrounded Cost ($)'].mean()
    peak = df['Unrounded Cost ($)'].max()
    anomaly_count = df['final_anomaly_flag'].sum() if 'final_anomaly_flag' in df.columns else 0
    
    return {
        "message": f"""**Cost Summary**

Total Cost: ${total:,.2f}
Average per Record: ${avg:,.2f}
Peak Cost: ${peak:,.2f}
Anomalies: {anomaly_count}"""
    }

def _get_top_services_response(df):
    if 'Service Name' not in df.columns:
        return {"message": "Service Name column not found."}
    
    top = df.groupby('Service Name')['Unrounded Cost ($)'].sum().nlargest(5).reset_index()
    top.columns = ['Service Name', 'Total Cost ($)']
    
    return {"message": "Top 5 Services by Cost", "data": top}

def _get_recommendations_response(df):
    if 'recommendation' not in df.columns or 'final_anomaly_flag' not in df.columns:
        return {"message": "Run detection first to get recommendations."}
    
    anomalies = df[df['final_anomaly_flag'] == 1]
    if anomalies.empty:
        return {"message": "No recommendations needed."}
    
    total_savings = anomalies['est_savings'].sum()
    
    return {
        "message": f"**Optimization Recommendations**\n\nEstimated Savings Potential: ${total_savings:,.2f}",
        "data": anomalies[['Service Name', 'recommendation', 'est_savings']].sort_values('est_savings', ascending=False).head(5)
    }

def _get_region_response(df):
    if 'Region/Zone' not in df.columns:
        return {"message": "Region/Zone column not found."}
    
    region_summary = df.groupby('Region/Zone')['Unrounded Cost ($)'].sum().sort_values(ascending=False).reset_index()
    region_summary.columns = ['Region/Zone', 'Total Cost ($)']
    
    return {"message": "Regional Cost Distribution", "data": region_summary}

def _get_trend_response(df):
    if 'start_date' not in df.columns:
        return {"message": "Date information not found."}
    
    trend = df.groupby('start_date')['Unrounded Cost ($)'].sum().reset_index()
    trend.columns = ['Date', 'Daily Cost ($)']
    trend = trend.sort_values('Date')
    
    fig = px.line(trend, x='Date', y='Daily Cost ($)', title='Daily Cost Trend')
    fig.update_layout(height=400, plot_bgcolor='white', title_x=0.5)
    fig.update_xaxes(gridcolor='#e5e5e5')
    fig.update_yaxes(gridcolor='#e5e5e5')
    
    return {"message": "Daily Cost Trend", "chart": fig}

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomalies", "Optimization", "ChatOps"])

with tab1:
    uploaded_file = st.file_uploader("Upload GCP Billing CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.caption(f"{len(df):,} records loaded")
        
        if st.session_state.engine is None:
            st.session_state.engine = IntelligentFinOpsEngine()
        
        with st.spinner("Running anomaly detection..."):
            results = st.session_state.engine.predict(df)
            st.session_state.results_df = results
        
        total_cost = results['Unrounded Cost ($)'].sum()
        anomaly_count = results['final_anomaly_flag'].sum()
        anomaly_rate = (anomaly_count / len(results)) * 100
        est_savings = results[results['final_anomaly_flag'] == 1]['est_savings'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='stat-card'><div class='metric-label'>Total Cost</div><div class='metric-value'>${total_cost:,.2f}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='stat-card'><div class='metric-label'>Anomalies</div><div class='metric-value'>{anomaly_count}</div><div class='metric-label'>{anomaly_rate:.1f}% of records</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='stat-card'><div class='metric-label'>Records</div><div class='metric-value'>{len(results):,}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='stat-card'><div class='metric-label'>Est. Savings</div><div class='metric-value'>${est_savings:,.2f}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        st.subheader("Daily Cost Trend")
        
        if 'start_date' in results.columns:
            daily_data = results.groupby('start_date').agg({
                'Unrounded Cost ($)': 'sum',
                'final_anomaly_flag': 'sum'
            }).reset_index()
            daily_data.columns = ['Date', 'Daily Cost', 'Anomaly Count']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=daily_data['Daily Cost'],
                name='Daily Cost',
                line=dict(color='#1a1a1a', width=2),
                fill='tozeroy',
                fillcolor='rgba(26, 26, 26, 0.05)'
            ))
            
            anomaly_days = daily_data[daily_data['Anomaly Count'] > 0]
            if not anomaly_days.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_days['Date'],
                    y=anomaly_days['Daily Cost'],
                    name='Anomaly',
                    mode='markers',
                    marker=dict(color='#dc3545', size=8, symbol='circle')
                ))
            
            daily_data['7-Day Avg'] = daily_data['Daily Cost'].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily_data['Date'],
                y=daily_data['7-Day Avg'],
                name='7-Day Average',
                line=dict(color='#666', width=1.5, dash='dash')
            ))
            
            fig.update_layout(
                height=450,
                plot_bgcolor='white',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            fig.update_xaxes(gridcolor='#e5e5e5', showgrid=True)
            fig.update_yaxes(gridcolor='#e5e5e5', showgrid=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Service Name' in results.columns:
            st.subheader("Cost by Service")
            service_cost = results.groupby('Service Name')['Unrounded Cost ($)'].sum().sort_values(ascending=False).head(8)
            
            fig_bar = px.bar(
                x=service_cost.values,
                y=service_cost.index,
                orientation='h',
                text=service_cost.values,
                labels={'x': 'Cost ($)', 'y': ''}
            )
            fig_bar.update_traces(textposition='outside', marker_color='#1a1a1a')
            fig_bar.update_layout(height=400, plot_bgcolor='white', showlegend=False)
            fig_bar.update_xaxes(gridcolor='#e5e5e5')
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("Upload a GCP billing CSV to begin analysis")
        st.markdown("""
        **Required columns:** `Usage Start Date`, `Unrounded Cost ($)`, `Service Name`, `Region/Zone`
        """)

with tab2:
    if st.session_state.results_df is not None:
        df = st.session_state.results_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Severity",
                options=['Critical', 'High', 'Medium', 'Low'],
                default=['Critical', 'High', 'Medium', 'Low']
            )
        
        filtered_df = df[df['final_anomaly_flag'] == 1]
        if severity_filter:
            filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
        
        anomaly_counts = {
            'Daily Cost Spike': df['daily_spike_flag'].sum() if 'daily_spike_flag' in df.columns else 0,
            'Service Deviation': df['service_deviation_flag'].sum() if 'service_deviation_flag' in df.columns else 0,
            'Region Spike': df['region_spike_flag'].sum() if 'region_spike_flag' in df.columns else 0,
            'Underutilized': df['underutilized_flag'].sum() if 'underutilized_flag' in df.columns else 0,
            'Usage Surge': df['usage_surge_flag'].sum() if 'usage_surge_flag' in df.columns else 0
        }
        
        anomaly_df = pd.DataFrame([
            {'Type': k, 'Count': v} for k, v in anomaly_counts.items() if v > 0
        ])
        
        if not anomaly_df.empty:
            fig = px.bar(
                anomaly_df,
                x='Type',
                y='Count',
                text='Count',
                labels={'Count': 'Occurrences'}
            )
            fig.update_traces(marker_color='#1a1a1a', textposition='outside')
            fig.update_layout(height=400, plot_bgcolor='white', xaxis_tickangle=-45)
            fig.update_xaxes(gridcolor='#e5e5e5')
            fig.update_yaxes(gridcolor='#e5e5e5')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detected Anomalies")
        
        if not filtered_df.empty:
            display_cols = ['severity', 'root_cause', 'Service Name', 'Region/Zone', 'Unrounded Cost ($)', 'recommendation']
            available_cols = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols].sort_values('Unrounded Cost ($)', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No anomalies match the selected filters")
    else:
        st.info("Upload a dataset in the Overview tab first")

with tab3:
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        anomalies = df[df['final_anomaly_flag'] == 1].copy()
        
        if not anomalies.empty:
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='stat-card'><div class='metric-label'>Opportunities</div><div class='metric-value'>{len(anomalies)}</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='stat-card'><div class='metric-label'>Est. Savings</div><div class='metric-value'>${anomalies['est_savings'].sum():,.2f}</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='stat-card'><div class='metric-label'>Avg Savings</div><div class='metric-value'>${anomalies['est_savings'].mean():,.2f}</div></div>", unsafe_allow_html=True)
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            st.subheader("Recommendations")
            
            rec_summary = anomalies.groupby('recommendation').agg({
                'est_savings': 'sum',
                'Unrounded Cost ($)': 'count'
            }).rename(columns={'Unrounded Cost ($)': 'count'}).sort_values('est_savings', ascending=False)
            
            for idx, row in rec_summary.iterrows():
                with st.expander(f"{idx} — ${row['est_savings']:,.2f} ({row['count']} resources)"):
                    st.markdown(f"Estimated savings: ${row['est_savings']:,.2f}")
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            if st.button("Run Simulation", type="primary", use_container_width=True):
                with st.spinner("Running simulation..."):
                    st.session_state.simulation_run = True
                    
                    st.markdown("#### Simulation Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Actions**")
                        if anomalies['underutilized_flag'].sum() > 0:
                            st.markdown(f"• Rightsize {anomalies['underutilized_flag'].sum()} underutilized resources")
                        if anomalies['daily_spike_flag'].sum() > 0:
                            st.markdown(f"• Optimize scaling for {anomalies['daily_spike_flag'].sum()} spike events")
                        if anomalies['service_deviation_flag'].sum() > 0:
                            st.markdown(f"• Reconfigure {anomalies['service_deviation_flag'].sum()} services")
                    
                    with col2:
                        savings_by_severity = anomalies.groupby('severity')['est_savings'].sum()
                        st.markdown("**Savings by Severity**")
                        for severity, amount in savings_by_severity.items():
                            st.markdown(f"• {severity}: ${amount:,.2f}")
                    
                    st.success(f"Total estimated savings: ${anomalies['est_savings'].sum():,.2f}")
        else:
            st.success("No anomalies detected")
    else:
        st.info("Upload a dataset in the Overview tab first")

with tab4:
    render_chatops()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.caption("GCP FinOps v3.0 | Isolation Forest + Rule-Based Detection")
