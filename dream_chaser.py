from langchain import OPENAI, LLMCHain, PromptTemplate
import config
import os

os.environ['OPENAI_API_KEY'] = config.OPENAI_KEY
