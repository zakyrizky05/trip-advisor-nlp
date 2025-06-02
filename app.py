import altair as alt
import eda
import inf
import streamlit as st

#define pages
PAGES = {
    # format 'judul menu' : nama_modul
    "EDA": eda,
    "Inference": inf
}

#set sidebar title
st.sidebar.title('Navigation')

# set sidebar selection
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))

# run app function in selected page
page = PAGES[selection]
page.app()