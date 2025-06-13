import streamlit as st
import os
import asyncio

async def main():

    if "file" not in st.session_state:
        st.session_state.file = None

    st.header("Image Tools")

    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.file = uploaded_file
            b = uploaded_file.getvalue()
            with open(uploaded_file.name, "wb") as f:
                f.write(b)

    with col2:
        if st.session_state.file is not None:
                st.image(uploaded_file)
        

if __name__ == "__main__":
    asyncio.run(main())