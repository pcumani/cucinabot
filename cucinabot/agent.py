import os
import mimetypes
import base64
import yaml
from typing import TypedDict, Annotated, Union
from dotenv import load_dotenv
from psycopg.errors import ConnectionTimeout

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

# Import our custom tools from their modules
from cucinabot.tools import unit_converter_tool, \
                multiply, add, subtract, divide, \
                get_user_info, save_user_info, delete_user_info
from cucinabot.retriever import recipes_info_tool

load_dotenv()

# Database connection string postgresql://USER:PASSWORD@EXTERNAL_HOST:PORT/DATABASE
try:
    DATABASE_URL = os.environ["DATABASE_URL"]
except KeyError:
    # default to a local database if not set
    DATABASE_URL = "postgresql://cucinabot:cucinabot@localhost:5432/postgres?sslmode=disable"

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class FinalAgent:
    
    def __init__(self, model_type: str="GOOGLE", system_prompt_path: str="system_prompt.yaml", use_memory: bool=True):
        """
        Args: model_type "GOOGLE" or "HUGGINGFACE" or "OLLAMA"
        """
        with open(system_prompt_path, 'r') as stream:
            self.system_prompt = yaml.safe_load(stream)
        
        self.model_type = model_type
        self.create_model(model_type, system_prompt=self.system_prompt, use_memory=use_memory)

        
    def create_model(self, model_type: str, system_prompt: Union[str, dict, None]=None, use_memory: bool=True) -> CompiledStateGraph:
        """Create the LLM model of the agent.
        Args:
            model_type (str): The type of model to use, e.g., "GOOGLE", "HUGGINGFACE", "OLLAMA".
            system_prompt (Union[str, dict, None]): The system prompt to use for the agent.
            use_memory (bool): Whether to use memory for the agent. It needs a running Postgres database.
        Returns:
            CompiledStateGraph: The compiled state graph of the agent.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        model_type = model_type.upper().replace(' ', '')
        if model_type not in ["GOOGLE", "OLLAMA", "HUGGINGFACE"]:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: GOOGLE, OLLAMA, HUGGING FACE.")

        self.model_type = model_type 

        if model_type == "HUGGINGFACE":
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
            # Initialize the Hugging Face model
            # Generate the chat interface, including the tools
            llm = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"
            )
            chat = ChatHuggingFace(llm=llm, verbose=True)
        elif model_type == "OLLAMA":
            from langchain_ollama import ChatOllama
            chat = ChatOllama(model = "qwen3:8b")
        elif model_type == "GOOGLE":
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.rate_limiters import InMemoryRateLimiter
            rate_limiter = InMemoryRateLimiter(
                        # Max allowed rate per free API: 10 requests per minute, but we use 6 to avoid hitting the limit on subsquent answers.
                        requests_per_second=6/60, 
                        # Wake up every 100 ms to check whether allowed to make a request,
                        check_every_n_seconds=0.1,
                        max_bucket_size=10,  # Controls the maximum burst size.
                    )
            chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", rate_limiter=rate_limiter)

        tools = [unit_converter_tool,
                multiply, add, subtract, divide,
                recipes_info_tool, get_user_info, save_user_info, delete_user_info]
        chat_with_tools = chat.bind_tools(tools)

        def assistant(state: AgentState):
            messages = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=1e6 if self.model_type == "GOOGLE" else 126000,
                start_on="human",
                end_on=("human", "tool"),
            )
            if isinstance(system_prompt, str):
                return {
                    "messages": [SystemMessage(content=system_prompt)] + messages,
                }
            elif isinstance(system_prompt, dict) and "system_prompt" in system_prompt:
                return {
                    "messages": [chat_with_tools.invoke([SystemMessage(content=system_prompt['system_prompt'])] + messages)],
                }
            else:
                print("No system prompt provided, using default.")
                return {
                    "messages": [chat_with_tools.invoke(messages)],
                }

        builder = StateGraph(AgentState)

        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        if use_memory:
            try:
                from psycopg_pool import ConnectionPool
                from psycopg.rows import dict_row

                pool = ConnectionPool(DATABASE_URL, kwargs={"row_factory": dict_row, "prepare_threshold": 0}, min_size=1, max_size=5)
                store = PostgresStore(pool)
                store.setup()

                checkpointer = PostgresSaver(pool)
                checkpointer.setup()
            except ConnectionTimeout as e:
                print(f"Error connecting to Postgres database ({e}).")
                raise e
            self.agent = builder.compile(checkpointer=checkpointer, store=store)

            # We want to always start with a clean memory for the agent
            try:
                self.clear_thread_memory()
            except Exception as e:
                print(f"No memory to clear at startup ({e}), continuing...")
                pass
        else:
            checkpointer = None
            store=None
            self.agent = builder.compile()
        print(f"Agent from {model_type} initialized.")
    
    def clear_thread_memory(self, thread_id: str='1') -> None:
        """ Clear the checkpointer memory for a given thread_id. """
        memory = self.agent.checkpointer
        if memory is None:
            return
        try:
            self.agent.checkpointer.delete_thread(thread_id)

            print(f"Memory cleared for thread_id: {thread_id}")
            return

        except Exception as e:
            print(f"Error clearing InMemorySaver storage for thread_id {thread_id}: {e}")
    
    
    def __call__(self, question: str, attached_file: dict, recursion_limit: int=-1, thread_id: str='1', user_id: str='user_0') -> str:
        """Invoke the agent with a question and an optional attached file.

        Args:
            question (str): The question to ask the agent.
            attached_file (dict): A dictionary of thet attached file with keys 'name', 'content', and 'path'.
                - 'name': The name of the attached file.
                - 'content': The content of the attached file as bytes.
                - 'path': Optional, The path to the attached file.
            recursion_limit (int): The recursion limit for the agent. Default is -1, which means no limit.
            thread_id (str): The thread ID for the conversation. Default is '1'.
            user_id (str): The user ID for the conversation. Default is 'user_0'.
        """
        print(f"Agent received question (first 100 chars): {question[:100]}...")

        if attached_file['name'] != "" and attached_file['content'] is not None:
            mime_type, _ = mimetypes.guess_type(attached_file['name'])
            if mime_type.startswith("image/") or mime_type.startswith("audio/") or mime_type.startswith("video/"):
                # Image file - convert to base64
                encoded_file = base64.b64encode(attached_file['content']).decode('utf-8')
                #
                if self.model_type == "GOOGLE":
                    question = [{"type": "text", "text": question},
                            {"type": "image" if mime_type.startswith("image/") else "media",
                             "source_type": "base64",
                             "data": encoded_file,
                             "mime_type": mime_type,},
                                ]
                else:
                    question = f"{question}\n\nAttached file extension:{attached_file['name'].split('.')[-1]} - Attached file base64 encoded: \n{encoded_file}"
            elif mime_type.startswith("text/"):
                # Text-based file (like .py, .txt, .json)
                question = f"{question}\n\nAttached file extension:{attached_file['name'].split('.')[-1]} - Attached file content: \n{attached_file['content'].decode('utf-8')}"
            else:
                encoded_file = base64.b64encode(attached_file['content']).decode('utf-8')
                print(f"Unsupported file {attached_file['name']} type: {mime_type}. Only images, audio, video, and text files are supported.")
                question = f"{question}\n\nAttached file extension: {attached_file['name'].split('.')[-1]}. File path: {attached_file['path']} - Attached file base64 encoded:\n{encoded_file}"

        if recursion_limit>0:
            return self.agent.invoke({"messages": [HumanMessage(content=question)]}, {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id, "user_id": user_id}})
        else:
            return self.agent.invoke({"messages": [HumanMessage(content=question)]}, {"configurable": {"thread_id": thread_id, "user_id": user_id}})


