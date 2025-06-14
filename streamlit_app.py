import streamlit as st
import os
import asyncio
from typing import Optional, Any, List, Union
from pydantic import BaseModel, Field, SecretStr
from pathlib import Path

def clear():
	st.session_state.file = None
	st.rerun()

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

from PIL import Image
import PIL.ImageFilter
import PIL.ImageEnhance
from rembg import remove

def init_mcp_sever():
	mcp = FastMCP("Image Handler")

	@mcp.tool()
	def flip_vertically() -> str:
		"""Flips image horizontally. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
		img.save("image.png")
		return "SUCCESS"

	@mcp.tool()
	def flip_horizontally() -> str:
		"""Flips image vertically. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
		img.save("image.png")
		return "SUCCESS"

	@mcp.tool()
	def rotate_90() -> str:
		"""Rotates image by 90 degrees. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.ROTATE_90)
		img.save("image.png")
		return "SUCCESS"

	mcp2 = FastMCP("Image Handler")

	@mcp2.tool()
	def roll(delta: int):
		"""Rolls image by amout of pixels provided. Image is provided on the external server. """
		img = Image.open("image.png")
		xsize, ysize = img.size
		delta = delta % xsize

		if delta == 0:
			img.save("image.png")
			return "SUCCESS"

		part1 = img.crop((0, 0, delta, ysize))
		part2 = img.crop((delta, 0, xsize, ysize))
		img.paste(part1, (xsize - delta, 0, xsize, ysize))
		img.paste(part2, (0, 0, xsize - delta, ysize))

		img.save("image.png")
		return "SUCCESS"

	@mcp2.tool()
	def monochrome():
		"""Converts image to monochrome scale. Image is provided on the external server. """
		img = Image.open("image.png")
		e = PIL.ImageEnhance.Color(img)
		img = e.enhance(0.0);
		img.save("image.png")
		return "SUCCESS"

	@mcp2.tool()
	def remove_background():
		"""Removes background from the image. Image is provided on the external server."""
		img = Image.open("image.png")
		img = remove(img)
		img.save("image.png")

	mcp3 = FastMCP("Image Handler")

	@mcp3.tool()
	def flip_vertically() -> str:
		"""Flips image horizontally. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
		img.save("image.png")
		return "SUCCESS"

	@mcp3.tool()
	def flip_horizontally() -> str:
		"""Flips image vertically. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
		img.save("image.png")
		return "SUCCESS"

	@mcp3.tool()
	def rotate_90() -> str:
		"""Rotates image by 90 degrees. Image is provided on the external server. """
		img = Image.open("image.png")
		img = img.transpose(Image.Transpose.ROTATE_90)
		img.save("image.png")
		return "SUCCESS"

	@mcp3.tool()
	def roll(delta: int):
		"""Rolls image by amout of pixels provided. Image is provided on the external server. """
		img = Image.open("image.png")
		xsize, ysize = img.size
		delta = delta % xsize

		if delta == 0:
			img.save("image.png")
			return "SUCCESS"

		part1 = img.crop((0, 0, delta, ysize))
		part2 = img.crop((delta, 0, xsize, ysize))
		img.paste(part1, (xsize - delta, 0, xsize, ysize))
		img.paste(part2, (0, 0, xsize - delta, ysize))

		img.save("image.png")
		return "SUCCESS"

	@mcp3.tool()
	def monochrome():
		"""Converts image to monochrome scale. Image is provided on the external server. """
		img = Image.open("image.png")
		e = PIL.ImageEnhance.Color(img)
		img = e.enhance(0.0);
		img.save("image.png")
		return "SUCCESS"

	@mcp3.tool()
	def remove_background():
		"""Removes background from the image. Image is provided on the external server."""
		img = Image.open("image.png")
		img = remove(img)
		img.save("image.png")

	if "mcp_version" not in st.session_state:
		return mcp

	if st.session_state.mcp_version == 1:
		return mcp
	elif st.session_state.mcp_version == 2:
		return mcp2
	elif st.session_state.mcp_version == 3:
		return mcp3

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
			if st.button("Version 1"):
				st.session_state.mcp_version = 1
				st.rerun()

			if st.button("Version 2"):
				st.session_state.mcp_version = 2
				st.rerun()

			if st.button("Version 3"):
				st.session_state.mcp_version = 3
				st.rerun()

			for tool in st.session_state.tools:
				st.markdown(f"**{tool.name}:**\n{tool.description}")
	
		with col1:
			if st.session_state.file is None:
				uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
				if uploaded_file is not None:
					st.session_state.file = uploaded_file
					b = uploaded_file.getvalue()
					n = f"image.{Path(uploaded_file.name).suffix}"
					with open(n, "wb") as f:
						f.write(b)
					img = Image.open(n)
					img.save("image.png")
					st.rerun()

			if(st.button("Clear")):
				clear()

			if prompt := st.chat_input("What shall I do?"):
				response = await agent.ainvoke(
					{
						"messages": [
							SystemMessage(content="You are an image handling service. Use provided tools to perform operations on the image. Image is provided by the externally and your job is only to invoke correct functions to modify the picture. With every answer try to use one of your tools! Call tools asynchronously!"),
							HumanMessage(content=prompt),
						]
					})
				st.text(response["messages"][-1].content)

		with col2:
			if st.session_state.file is not None:
				st.image("image.png")

if __name__ == "__main__":
	asyncio.run(main())