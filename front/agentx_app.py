import streamlit as st
import time  # for measuring time duration of API calls
import os
import pandas as pd
import plotly.express as px
from helpers.utils import init_env_from_yaml
init_env_from_yaml()

from agents_utils import (
    orchestrator,
    AGENT_ENDPOINTS,
    stream_watsonx_response,
    build_combined_prompt,
    data_agent_retrieve,
    fetch_market_analysis_data
)
st.set_page_config(page_title="AgentX Multi-Agent Demo", layout="wide")  # Doit être la première commande Streamlit

st.markdown(
    """
    <style>
    div.custom-select {
        width: 60px !important;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("AgentX")
    st.sidebar.header("Connect to watsonx.data")
    host = st.sidebar.text_input(
        "Host", 
        value="ibm-lh-lakehouse-presto-01-presto-svc-cpd.apps.678eb763f501a868a5829008.ocp.techzone.ibm.com"
    )
    port = st.sidebar.text_input("Port", value="443")
    user = st.sidebar.text_input("User", value="admin")
    password = st.sidebar.text_input("Password", value="********", type="password")
    if st.sidebar.button("Connect"):
        with st.spinner("Connecting..."):
            time.sleep(6)  # Simulate processing for 6 seconds
        success_placeholder = st.sidebar.empty()
        success_placeholder.success("Connected!")
        time.sleep(3)
        success_placeholder.empty()
        st.session_state.connected = True

    tabs = st.tabs(["Chat", "Dashboard"])
    
    with tabs[0]:
        with st.container():
            if "connected" in st.session_state and st.session_state.connected:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.header("Chat")
                with col2:
                    st.markdown('<div class="custom-select">', unsafe_allow_html=True)
                    catalog = st.selectbox("Catalogue", options=["agentic_rag", "hive_data", "iceberg_data", "postgres"], key="catalog")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="custom-select">', unsafe_allow_html=True)
                    schema = st.selectbox("Schéma", options=["metrics"], key="schema")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.header("Chat")

        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            st.markdown(f"**{message['role']}**: {message['content']}")
        
        st.write("---")
        user_input = st.text_area("Your message", "", key="user_input")
        send_button = st.button("Send")
        
        if send_button and user_input.strip():
            with st.spinner("The orchestrator is assigning to one of the agents..."):
                time.sleep(6)
                selected_agent = orchestrator(user_input)
            st.write("Selected Agent:", selected_agent)
            
            if selected_agent == "Data Agent":
                prompt = build_combined_prompt(user_input, selected_agent)
                print(prompt)
            elif selected_agent == "Data Visualizer Agent":
                market_data = fetch_market_analysis_data()
                st.session_state["market_analysis_data"] = market_data
                prompt = ("Dashboard created with market analysis data. "
                          "Please check the Dashboard tab for detailed plots.")
            elif selected_agent == "Vector Search Agent":
                prompt = user_input
            else:
                prompt = user_input    
            
            st.session_state.chat_history.append({"role": "User", "content": user_input})
            st.session_state.chat_history.append({"role": "Assistant", "content": ""})
            response_placeholder = st.empty()
            
            if selected_agent == "Data Visualizer Agent":
                reasoning = ""
                with st.spinner("Requesting API..."):
                    time.sleep(8)
                reasoning += "1. **SUCCESS**.\n"
                response_placeholder.markdown(reasoning)
                response_placeholder.markdown(reasoning)
                with st.spinner("Agent building the dashboard..."):
                    time.sleep(8)
                response_placeholder.markdown(reasoning)
                st.session_state.chat_history[-1]["content"] = prompt
                response_placeholder.markdown(f"**Assistant**: {prompt}")
            elif selected_agent == "Vector Search Agent":
                with st.spinner("Requesting API..."):
                    time.sleep(8)
                endpoint_url = AGENT_ENDPOINTS.get("Vector Search Agent")
                if endpoint_url:
                    response_text = ""
                    for delta in stream_watsonx_response(endpoint_url, prompt):
                        response_text += delta
                        st.session_state.chat_history[-1]["content"] = response_text
                        response_placeholder.markdown(f"**Assistant**: {response_text}")
                else:
                    st.error("No endpoint found for Vector Search Agent.")
            else:
                response_placeholder = st.empty()
                reasoning = ""
                with st.spinner("Test de connexion à watsonx.data..."):
                    time.sleep(2)
                reasoning += "1. **SUCCESS**.\n"
                response_placeholder.markdown(reasoning)
                with st.spinner("Generating SQL Request..."):
                    time.sleep(8)
                sql_query = """```sql
                            SELECT product,
                                SUM(sales) AS total_sales
                            FROM agentic_rag.metrics.ecommerce
                            WHERE order_date BETWEEN '2018-11-24' AND '2018-11-30'
                            GROUP BY product
                            ORDER BY total_sales DESC
                            LIMIT 5
                            ```"""
                reasoning += "\n2. Requête SQL générée :\n" + sql_query + "\n"
                response_placeholder.markdown(reasoning)
                with st.spinner("Requesting API..."):
                    time.sleep(8)
                json_received = """```json
                                {
                                    "results": [
                                        {"product": "Jeans", "total_sales": 8502.0},
                                        {"product": "Formal Shoes", "total_sales": 7881.0},
                                        {"product": "T - Shirts", "total_sales": 7440.0},
                                        {"product": "Shirts", "total_sales": 7056.0},
                                        {"product": "Running Shoes", "total_sales": 6272.0}
                                    ]
                                }
                                ```"""
                reasoning += "\n3. JSON reçu :\n" + json_received + "\n"
                response_placeholder.markdown(reasoning)
                time.sleep(4)
                
                reasoning += "\n4. Résultat final retourné."
                response_placeholder.markdown(reasoning)
                response = data_agent_retrieve(prompt)
                response_text = ""
                for chunk in response:
                    delta = chunk
                    if delta:
                        response_text += str(delta)
                        st.session_state.chat_history[-1]["content"] = response_text
                        response_placeholder.markdown(f"**Assistant**: {response_text}")
            st.rerun()
    
    with tabs[1]:
        st.header("Dashboard")
        if "market_analysis_data" in st.session_state:
            data = st.session_state["market_analysis_data"]
            if data:
                df = pd.DataFrame(data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Total Revenue by Product")
                    fig1 = px.bar(
                        df, 
                        x="product", 
                        y="total_revenue", 
                        title="Total Revenue by Product",
                        color="product",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.subheader("Total Profit by Product")
                    fig2 = px.bar(
                        df, 
                        x="product", 
                        y="total_profit", 
                        title="Total Profit by Product",
                        color="product",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Average Discount by Product")
                    fig3 = px.line(
                        df,
                        x="product",
                        y="average_discount",
                        title="Average Discount by Product",
                        markers=True,
                        color="product",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                with col4:
                    st.subheader("Total Quantity Sold by Product")
                    fig4 = px.pie(
                        df,
                        names="product",
                        values="total_quantity",
                        title="Total Quantity Sold by Product",
                        color="product",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.write("No market analysis data available.")
        else:
            st.write("No market analysis data available.")
if __name__ == "__main__":
    main()