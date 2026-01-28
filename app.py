import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide", page_icon="📊")

# Title
st.title("🎯 Customer Retention & Churn Prediction Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/customer_segments.csv')
    return df

# Load model
@st.cache_resource
def load_model():
    try:
        with open('outputs/models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('outputs/models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

try:
    df = load_data()
    model, scaler = load_model()
    
    # Sidebar
    st.sidebar.header("🔍 Filters")
    
    # Segment filter
    segment_options = df['Segment_Name'].unique().tolist()
    segments = st.sidebar.multiselect(
        "Select Customer Segments",
        options=segment_options,
        default=segment_options
    )
    
    # Churn filter
    churn_filter = st.sidebar.radio(
        "Customer Status",
        ["All", "Active Only", "Churned Only"]
    )
    
    # Filter data
    filtered_df = df[df['Segment_Name'].isin(segments)].copy()
    if churn_filter == "Active Only":
        filtered_df = filtered_df[filtered_df['Churned'] == 0]
    elif churn_filter == "Churned Only":
        filtered_df = filtered_df[filtered_df['Churned'] == 1]
    
    # KPIs
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    with col2:
        churn_rate = filtered_df['Churned'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        total_revenue = filtered_df['Monetary'].sum()
        st.metric("Total Revenue", f"£{total_revenue:,.0f}")
    with col4:
        avg_value = filtered_df['Monetary'].mean()
        st.metric("Avg Customer Value", f"£{avg_value:.2f}")
    
    st.markdown("---")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Customer Segments Distribution")
        segment_counts = filtered_df['Segment_Name'].value_counts()
        fig1 = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customers by Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue by Segment")
        segment_revenue = filtered_df.groupby('Segment_Name')['Monetary'].sum().sort_values(ascending=False)
        fig2 = px.bar(
            x=segment_revenue.index,
            y=segment_revenue.values,
            title="Total Revenue by Segment",
            labels={'x': 'Segment', 'y': 'Revenue (£)'},
            color=segment_revenue.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # RFM Analysis
    st.header("🔍 RFM Distribution Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig3 = px.histogram(
            filtered_df, 
            x='Recency', 
            title='Recency Distribution',
            labels={'Recency': 'Days Since Last Purchase'},
            nbins=30,
            color_discrete_sequence=['#636EFA']
        )
        fig3.add_vline(x=90, line_dash="dash", line_color="red", 
                       annotation_text="Churn Threshold (90 days)")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.histogram(
            filtered_df, 
            x='Frequency', 
            title='Frequency Distribution',
            labels={'Frequency': 'Number of Orders'},
            nbins=30,
            color_discrete_sequence=['#EF553B']
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col3:
        # Use log scale for monetary since it's skewed
        fig5 = px.histogram(
            filtered_df, 
            x='Monetary', 
            title='Monetary Distribution',
            labels={'Monetary': 'Total Spend (£)'},
            nbins=30,
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")
    
    # Churn Prediction Tool (only if model loaded)
    if model is not None and scaler is not None:
        st.header("🎯 Predict Customer Churn")
        st.markdown("Enter customer attributes to predict churn probability:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input("Recency (days)", min_value=0, max_value=400, value=30, 
                                     help="Days since last purchase")
            frequency = st.number_input("Frequency (orders)", min_value=1, max_value=200, value=5,
                                       help="Total number of orders")
        
        with col2:
            monetary = st.number_input("Monetary (£)", min_value=0.0, max_value=300000.0, value=1000.0,
                                      help="Total amount spent")
            avg_order = st.number_input("Avg Order Value (£)", min_value=0.0, value=200.0,
                                       help="Average spend per order")
        
        with col3:
            avg_days_between = st.number_input("Avg Days Between Purchases", min_value=0.0, value=30.0,
                                              help="Average gap between orders")
            is_one_time = st.selectbox("Is One-Time Buyer?", [0, 1], 
                                      format_func=lambda x: "Yes" if x==1 else "No")
        
        if st.button("🔮 Predict Churn Probability", type="primary"):
            # Prepare features
            features = np.array([[recency, frequency, monetary, avg_order, avg_days_between, is_one_time]])
            features_scaled = scaler.transform(features)
            
            # Predict
            churn_prob = model.predict_proba(features_scaled)[0][1]
            
            # Display result
            st.markdown("### 📊 Prediction Result:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Churn Probability", f"{churn_prob*100:.1f}%", 
                         delta=f"{(churn_prob-0.423)*100:.1f}% vs avg", 
                         delta_color="inverse")
            
            with col2:
                if churn_prob > 0.7:
                    st.error("🚨 HIGH RISK - Immediate intervention needed!")
                elif churn_prob > 0.5:
                    st.warning("⚠️ MODERATE RISK - Monitor closely")
                else:
                    st.success("✅ LOW RISK - Customer is healthy")
            
            # Recommendation
            st.markdown("### 💡 Recommended Action:")
            if churn_prob > 0.7:
                if monetary > 2000:
                    st.info("📞 **High-Value Customer**: Personal outreach from account manager + VIP perks")
                else:
                    st.info("📧 **Standard Intervention**: Send 20% discount code + product recommendations")
            elif churn_prob > 0.5:
                st.info("📧 **Early Warning**: Re-engagement email with personalized offers")
            else:
                st.info("👍 **Healthy Customer**: Continue normal customer journey")
            
            # Visual gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob * 100,
                title = {'text': "Churn Risk Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if churn_prob > 0.7 else "orange" if churn_prob > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning("⚠️ Churn prediction model not loaded. Make sure model files exist in outputs/models/")
    
    st.markdown("---")
    
    # At-Risk Customers Table
    st.header("⚠️ Top 20 At-Risk Customers")
    
    at_risk = filtered_df[filtered_df['Churned'] == 1].sort_values('Monetary', ascending=False).head(20)
    
    if len(at_risk) > 0:
        # Format the dataframe for display
        display_df = at_risk[['CustomerID', 'Segment_Name', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue']].copy()
        display_df['Monetary'] = display_df['Monetary'].apply(lambda x: f"£{x:,.2f}")
        display_df['AvgOrderValue'] = display_df['AvgOrderValue'].apply(lambda x: f"£{x:,.2f}")
        display_df.columns = ['Customer ID', 'Segment', 'Days Since Purchase', 'Total Orders', 'Total Spent', 'Avg Order Value']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = at_risk.to_csv(index=False)
        st.download_button(
            label="📥 Download Full At-Risk Customers List",
            data=csv,
            file_name="at_risk_customers.csv",
            mime="text/csv"
        )
    else:
        st.success("🎉 No at-risk customers in current filter!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Customer Retention & Churn Analysis Dashboard</p>
        <p>Built with Streamlit • Data Science Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Error loading dashboard: {e}")
    st.info("💡 Make sure you've run the analysis notebooks first to generate required files!")
    
    with st.expander("🔍 Debug Information"):
        import os
        st.write("Current directory:", os.getcwd())
        st.write("Files check:")
        st.write("- customer_segments.csv:", os.path.exists('data/processed/customer_segments.csv'))
        st.write("- churn_model.pkl:", os.path.exists('outputs/models/churn_model.pkl'))
        st.write("- scaler.pkl:", os.path.exists('outputs/models/scaler.pkl'))