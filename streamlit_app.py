import streamlit as st
import os
import asyncio
from PIL import Image

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
        
            img = Image.open(uploaded_file.name)

            if(st.button("Flip Horizontally")):
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                img.save(uploaded_file.name)

    with col2:
        if st.session_state.file is not None:
            st.image(uploaded_file)
        

if __name__ == "__main__":
    asyncio.run(main())