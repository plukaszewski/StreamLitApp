import streamlit as st
import os
import asyncio

async def main():
    st.caption("Image Tools")

    

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        b = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(b)

    st.image(uploaded_file)

if __name__ == "__main__":
    asyncio.run(main())