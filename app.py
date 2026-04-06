import streamlit as st

st.set_page_config(page_title="ML Explorer", layout="wide")

app_choice = st.sidebar.radio("Choose App", ["Decision Tree", "Neural Network"], key="app_selector")

if app_choice == "Decision Tree":
    from decision_tree_app import run
    run()
else:
    from neural_network_app import run
    run()
