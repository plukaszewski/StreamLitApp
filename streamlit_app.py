import streamlit as st
import os
import asyncio
from typing import Optional, Any, List, Union
from pydantic import BaseModel, Field, SecretStr

##########IMG##########
from PIL import Image

a = 0
fname = ""

def clear():
	st.session_state.file = None
	global fname
	fname = ""
	st.rerun()

def flip_vertically() -> str:
	global fname
	if fname != "":
		img = Image.open(fname)
		img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
		img.save(fname)
		return "SUCCESS"
	return "FAIL"

#######################

##########TEST#########

def test() -> str:
	global a
	a += 1
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
	selected_model = "deepseek/deepseek-chat-v3-0324:free"
	model = ChatOpenRouter(model_name = selected_model)

	tools = [
		StructuredTool.from_function(
			name="test",
			func=test,
			description="Tool useful to test if service is working",),
		StructuredTool.from_function(
			name="flip vertically",
			func=flip_vertically,
			description="Flips image vertically. Image is provided on the external server. ",)
	]

	agent = create_react_agent(model, tools)

	st.session_state.agent = agent


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
			fname = uploaded_file.name
			b = uploaded_file.getvalue()
			with open(uploaded_file.name, "wb") as f:
				f.write(b)
			st.rerun()

	if(st.button("Flip Vertically")):
		flip_vertically()

	if(st.button("Clear")):
		clear()

	if(st.button("TEST")):
		response = st.session_state.agent.invoke(
			{
				"messages": [
					SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools!"),
					HumanMessage(content="whats the weather in sf?"),

				]
			})
		st.text(response["messages"][-1].content)

	if(st.button("TEST2")):
		response = st.session_state.agent.invoke(
			{
				"messages": [
					SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools!"),
					HumanMessage(content="Flip the image vertically"),

				]
			})
		st.text(response["messages"][-1].content)

	if(st.button("TEST3")):
		response = st.session_state.agent.invoke(
			{
				"messages": [
					SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools!"),
					HumanMessage(content="Test if service is working"),

				]
			})
		st.text(response["messages"][-1].content)

	if(st.button("TEST4")):
		test()



with col2:
	if st.session_state.file is not None:
		st.image(st.session_state.file.name)

	st.text(a)