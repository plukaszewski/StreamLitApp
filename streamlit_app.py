import streamlit as st
import asyncio

async def main():
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.file = uploaded_file

if __name__ == "__main__":
    asyncio.run(main())