import streamlit as st
import os
import asyncio
from typing import Optional, Any, List, Union
from pydantic import BaseModel, Field, SecretStr

##########IMG##########
from PIL import Image

def clear():
	st.session_state.file = None
	st.rerun()

def flip_vertically() -> str:
	img = Image.open("image.jpg")
	img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
	img.save("image.jpg")
	return "SUCCESS"

def flip_horizontally() -> str:
	img = Image.open("image.jpg")
	img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
	img.save("image.jpg")
	return "SUCCESS"

def rotate_90() -> str:
	img = Image.open("image.jpg")
	img = img.transpose(Image.Transpose.ROTATE_90)
	img.save("image.jpg")
	return "SUCCESS"

def roll(delta: int):
	img = Image.open("image.jpg")
	xsize, ysize = img.size
	delta = delta % xsize

	if delta == 0:
		img.save("image.jpg")
		return "SUCCESS"

	part1 = img.crop((0, 0, delta, ysize))
	part2 = img.crop((delta, 0, xsize, ysize))
	img.paste(part1, (xsize - delta, 0, xsize, ysize))
	img.paste(part2, (0, 0, xsize - delta, ysize))

	img.save("image.jpg")
	return "SUCCESS"

#######################


##########TEST#########

def test() -> str:
	return "TEST SUCCESSFULL"

#######################


##########MCP##########
from fastmcp import Client, FastMCP
from mcp.types import (
	EmbeddedResource,
	ImageContent,
	CallToolResult,
	TextContent
)
NonTextContent = ImageContent | EmbeddedResource
#from langchain_mcp_adapters.tools import _convert_call_tool_result

def init_mcp_sever():
	mcp = FastMCP("Image Handler")

	@mcp.tool()
	def flip_vertically() -> str:
		"""Flips image horizontally. Image is provided on the external server. """
		img = Image.open("image.jpg")
		img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
		img.save("image.jpg")
		return "SUCCESS"

	@mcp.tool()
	def flip_horizontally() -> str:
		"""Flips image vertically. Image is provided on the external server. """
		img = Image.open("image.jpg")
		img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
		img.save("image.jpg")
		return "SUCCESS"

	@mcp.tool()
	def rotate_90() -> str:
		"""Rotates image by 90 degrees. Image is provided on the external server. """
		img = Image.open("image.jpg")
		img = img.transpose(Image.Transpose.ROTATE_90)
		img.save("image.jpg")
		return "SUCCESS"

	@mcp.tool()
	def roll(delta: int):
		"""Rolls image by amout of pixels provided. Image is provided on the external server. """
		img = Image.open("image.jpg")
		xsize, ysize = img.size
		delta = delta % xsize

		if delta == 0:
			img.save("image.jpg")
			return "SUCCESS"

		part1 = img.crop((0, 0, delta, ysize))
		part2 = img.crop((delta, 0, xsize, ysize))
		img.paste(part1, (xsize - delta, 0, xsize, ysize))
		img.paste(part2, (0, 0, xsize - delta, ysize))

		img.save("image.jpg")
		return "SUCCESS"

	return mcp

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
	




async def init_model():
	async with Client(init_mcp_sever()) as client:

		mcp_tools = await client.list_tools()

		def convert_tool(c, mcptool):

			async def call_tool(
				**arguments: dict[str, Any],
				) -> tuple[str | list[str], list[NonTextContent] | None]:
				call_tool_result = await c.call_tool(mcptool.name, arguments)
				return _convert_call_tool_result(call_tool_result)

			return StructuredTool(
			name=mcptool.name,
			description=mcptool.description or "",
			args_schema=mcptool.inputSchema,
			coroutine=call_tool,
			response_format="content_and_artifact",
			metadata=mcptool.annotations.model_dump() if mcptool.annotations else None,
			)

		tools = [
			StructuredTool.from_function(
				name="test",
				func=test,
				description="Tool useful to test if service is working",),
			StructuredTool.from_function(
				name="flip horizontally",
				func=flip_horizontally,
				description="Flips image horizontally. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="flip vertically",
				func=flip_vertically,
				description="Flips image vertically. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="rotate 90",
				func=rotate_90,
				description="Rotates image by 90 degrees. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="roll",
				func=roll,
				description="Rolls image by amout of pixels provided. Image is provided on the external server. ",),
		]

		tools = [convert_tool(client, t) for t in mcp_tools]

		st.session_state.tools = tools
		st.session_state.mcp_tools = mcp_tools

		selected_model = "deepseek/deepseek-chat-v3-0324:free"
		model = ChatOpenRouter(model_name = selected_model)
		agent = create_react_agent(model, tools)

		st.session_state.agent = agent
	

def _convert_call_tool_result(
	call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
	text_contents: list[TextContent] = []
	non_text_contents = []
	for content in call_tool_result:
		if isinstance(content, TextContent):
			text_contents.append(content)
		else:
			non_text_contents.append(content)

	tool_content: str | list[str] = [content.text for content in text_contents]
	if not text_contents:
		tool_content = ""
	elif len(text_contents) == 1:
		tool_content = tool_content[0]

	#if call_tool_result.isError:
	#    raise ToolException(tool_content)

	return tool_content, non_text_contents or None


async def main():
	async with Client(init_mcp_sever()) as client:

		mcp_tools = await client.list_tools()

		def convert_tool(c, mcptool):

			async def call_tool(
				**arguments: dict[str, Any],
				) -> tuple[str | list[str], list[NonTextContent] | None]:
				call_tool_result = await c.call_tool(mcptool.name, arguments)
				return _convert_call_tool_result(call_tool_result[-1])

			return StructuredTool(
			name=mcptool.name,
			description=mcptool.description or "",
			args_schema=mcptool.inputSchema,
			coroutine=call_tool,
			response_format="content_and_artifact",
			metadata=mcptool.annotations.model_dump() if mcptool.annotations else None,
			)

		tools = [
			StructuredTool.from_function(
				name="test",
				func=test,
				description="Tool useful to test if service is working",),
			StructuredTool.from_function(
				name="flip horizontally",
				func=flip_horizontally,
				description="Flips image horizontally. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="flip vertically",
				func=flip_vertically,
				description="Flips image vertically. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="rotate 90",
				func=rotate_90,
				description="Rotates image by 90 degrees. Image is provided on the external server. ",),
			StructuredTool.from_function(
				name="roll",
				func=roll,
				description="Rolls image by amout of pixels provided. Image is provided on the external server. ",),
		]

		tools = [convert_tool(client, t) for t in mcp_tools]

		st.session_state.tools = tools
		st.session_state.mcp_tools = mcp_tools

		selected_model = "deepseek/deepseek-chat-v3-0324:free"
		model = ChatOpenRouter(model_name = selected_model)
		agent = create_react_agent(model, tools)

		st.session_state.agent = agent

		if "file" not in st.session_state:
				st.session_state.file = None

		st.header("Image Tools")

		col1, col2 = st.columns(2)

		with st.sidebar:
			for tool in st.session_state.tools:
				st.text(tool.name)
	
		with col1:
			if st.session_state.file is None:
				uploaded_file = st.file_uploader("Choose a file", type=["jpg"])
				if uploaded_file is not None:
					st.session_state.file = uploaded_file
					fname = uploaded_file.name
					b = uploaded_file.getvalue()
					with open("image.jpg", "wb") as f:
						f.write(b)
					st.rerun()

			if(st.button("Flip Vertically")):
				flip_vertically()

			if(st.button("Clear")):
				clear()

			if(st.button("V")):
				response = agent.invoke(
					{
						"messages": [
							SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools!"),
							HumanMessage(content="Flip the image vertically"),

						]
					})
				st.text(response["messages"][-1].content)
				
			if(st.button("T")):
				for tool in st.session_state.mcp_tools:
					st.text(tool)

			if(st.button("C")):
				d = dict()
				await st.session_state.tools[0].arun(d)

			if prompt := st.chat_input("What shall I do?"):
				response = agent.invoke(
					{
						"messages": [
							SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools!"),
							HumanMessage(content=prompt),
						]
					})
				st.text(response["messages"][-1].content)

		with col2:
			if st.session_state.file is not None:
				st.image("image.jpg")

if __name__ == "__main__":
	asyncio.run(main())