#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Optional

from cucinabot.agent import FinalAgent

agent_class = FinalAgent()

def stream_to_gradio(
    agent,
    task: str,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    import gradio as gr

    #agent_reply = agent.agent.invoke({"messages": [HumanMessage(content=task)]}, {"configurable": {"thread_id": "1"}}, stream=True, additional_args=additional_args)
    agent_reply = agent(task, {'name': "", 'content': None})
    print(f'Last message of {len(agent_reply['messages'])} metadata {agent_reply['messages'][-1].response_metadata} {agent_reply['messages'][-1].usage_metadata}')
    if agent_reply['messages'][-1].response_metadata['finish_reason'] != 'STOP':
        print(agent_reply['messages'])

    yield gr.ChatMessage(role="assistant", content=str(agent_reply['messages'][-1].content))


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        global agent_class
        self.agent_class = agent_class
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def change_model_type(self, model_type: str) -> None:
        """Change the model type of the agent."""
        model_type = model_type.upper().replace(' ', '')
        if model_type not in ["GOOGLE", "OLLAMA", "HUGGING FACE"]:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: GOOGLE, OLLAMA, HUGGING FACE.")
        self.agent_class.create_model(model_type)
        print(f"Model type changed to: {self.agent_class.model_type}")


    def interact_with_agent(self, prompt, messages):
        import gradio as gr
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent_class, task=prompt):
            messages.append(msg)
            yield messages
        yield messages

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input,
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="CucinaBot",
                type="messages",
                avatar_images=(
                    None,
                    "./assets/logo.png",
                ),
                resizable=True,
                scale=2,
            )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

            with gr.Row(elem_classes="main-content", equal_height=True):
                clear_btn = gr.Button("🧹 Clear History", variant="secondary", elem_classes="btn btn-secondary")
                model_selector = gr.Dropdown(choices=["Google", "Ollama", "Hugging Face"], label="Inference provider", info="Which model provider should be used for inference? (Conversation history will be erased)", interactive=True)

            clear_btn.click(
                fn=self.agent_class.clear_thread_memory,
                outputs=[chatbot],
                show_progress=False
                )
            
            model_selector.change(
                fn=self.agent_class.create_model,
                inputs=[model_selector],
                outputs=[chatbot],
            )
        
        demo.launch(debug=True, share=False, **kwargs)


if __name__ == "__main__":
    # Launch the Gradio UI
    GradioUI().launch(server_name="0.0.0.0",
        server_port=7860,
        show_error=True)
