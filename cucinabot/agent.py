import mimetypes
import base64
import yaml
from typing import TypedDict, Annotated, Union
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

# Import our custom tools from their modules
from cucinabot.tools import unit_converter_tool, \
                multiply, add, subtract, divide
from cucinabot.retriever import recipes_info_tool

load_dotenv()

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
        """Change the model type of the agent."""

        try:
            self.clear_memory()
        except Exception:
            pass

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
                recipes_info_tool]
        chat_with_tools = chat.bind_tools(tools)

        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

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
            checkpointer = InMemorySaver()
            self.agent = builder.compile(checkpointer=checkpointer)
        else:
            checkpointer = None
            self.agent = builder.compile()
        print(f"Agent from {model_type} initialized.")
    
    def clear_memory(self, thread_id: str='1') -> None:
        """ Clear the memory for a given thread_id. """
        memory = self.agent.checkpointer
        if memory is None:
            return
        try:
            # If it's an InMemorySaver (which MemorySaver is an alias for),
            # we can directly clear the storage and writes
            if hasattr(memory, 'storage') and hasattr(memory, 'writes'):
                # Clear all checkpoints for this thread_id (all namespaces)
                memory.storage.pop(thread_id, None)

                # Clear all writes for this thread_id (for all namespaces)
                keys_to_remove = [key for key in memory.writes.keys() if key[0] == thread_id]
                for key in keys_to_remove:
                    memory.writes.pop(key, None)

                print(f"Memory cleared for thread_id: {thread_id}")
                return

        except Exception as e:
            print(f"Error clearing InMemorySaver storage for thread_id {thread_id}: {e}")
    
    def __call__(self, question: str, attached_file: dict, recursion_limit=-1) -> str:
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
            return self.agent.invoke({"messages": [HumanMessage(content=question)]}, {"recursion_limit": recursion_limit, "configurable": {"thread_id": "1"}})
        else:
            return self.agent.invoke({"messages": [HumanMessage(content=question)]}, {"configurable": {"thread_id": "1"}})


