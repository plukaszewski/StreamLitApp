import streamlit as st
import os
import asyncio
from typing import Optional, Any, List, Union
from pydantic import BaseModek, Field, SecretStr

##########IMG##########
from PIL import Image

def clear():
    st.session_state.file = None
    st.rerun()

def flip_vertically():
    if st.session_state.file is not None:
        img = Image.open(st.session_state.file.name)
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img.save(st.session_state.file.name)

#######################

##########TEST#########

def test(_):
    if "test" not in st.session_state:
        st.session_state.test = 0

    st.session_state.test += 1
    return "TEST SUCCESSFULL"

#######################

##########LLM##########
from langchain_openai import ChatOpenAI
from langchain.agents import Tool

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import create_react_agent

class ChatOpenRouter(ChatOpenAI):
        openai_api_key: Optional[SecretStr] = Field(
            alias="api_key", default_factory=st.secrets["API_KEY"]
        )
        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"openai_api_key": st.secrets["API_KEY"]}

        def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
            openai_api_key = openai_api_key or st.secrets["API_KEY"]
            super().__init__(base_url=st.secrets["BASE_URL"], openai_api_key=openai_api_key, **kwargs)

def init_model():
    selected_model = "mistralai/devstral-small:free"
    model = ChatOpenRouter(model_name = selected_model)

    tools = [
        StructuredTool.from_function(
            name = "test",
            func=test,
            description="Tool useful to test if service is working",
            return_direct=True,
            args_schema=BaseModel()),
        StructuredTool.from_function(
            name = "flip vertically",
            func=flip_vertically,
            description="Flips image vertically",
            return_direct=False,
            args_schema=BaseModel())
    ]

    agent = create_react_agent(model, tools)

    st.session_state.agent = agent




async def main():

    if "file" not in st.session_state:
        st.session_state.file = None

    if "agent" not in st.session_state:
        init_model()

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

        if(st.button("Flip Vertically")):
            flip_vertically()

        if(st.button("Clear")):
            clear()

        if(st.button("TEST")):
            for step in agent_executor.stream(
                {
                    "messages": [
                        SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image."),
                        HumanMessage(content="Flip image vertically"),

                    ]
                },
                stream_mode="values",
            ):
                st.text(step["messages"][-1].content)


    with col2:
        if st.session_state.file is not None:
            st.image(st.session_state.file.name)

        if "test"  in st.session_state:
            st.text(st.session_state.test)
        

if __name__ == "__main__":
    asyncio.run(main())