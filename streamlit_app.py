import streamlit as st
import os
import asyncio
from PIL import Image

def clear():
    st.session_state.file = None
    st.rerun()

def flip_vertically():
    if st.session_state.file is not None: 
        st.text("t")
        img = Image.open(st.session_state.file.name)
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img.save(st.session_state.file.name)
        st.rerun()


async def main():

    if "file" not in st.session_state:
        st.session_state.file = None

    st.header("Image Tools")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.file is None:
            uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                st.session_state.file = uploaded_file
                b = uploaded_file.getvalue()
                with open(uploaded_file.name, "wb") as f:
                    f.write(b)
                st.rerun()

        if(st.button("Flip Horizontally")):
            flip_vertically()

        if(st.button("Clear")):
            clear()

    with col2:
        if st.session_state.file is not None:
            st.image(st.session_state.file)
        

if __name__ == "__main__":
    asyncio.run(main())