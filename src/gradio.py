import gradio as gr
from src.retriever import RAG
from src.config import Config

def chat_interface(user_input, history, selected_course):
    response = RAG.get_response(user_input, history, selected_course)
    history.append((user_input, response))
    
    return "", history

def launch_ui():
    with gr.Blocks(title="RAG QA Chatbot") as demo:
        gr.Markdown("# 📚 UM Faculty Handbook QA Bot\nAsk questions about your faculty handbook for the 2024/2025 academic session.")

        with gr.Row():
            course_selector = gr.Dropdown(
                choices=Config.COURSES,
                label="Select Your Course",
                interactive=True,
        )
            
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Hi! How can I assist you today?", show_label=False)

        state = gr.State([])  # To keep track of chat history

        user_input.submit(chat_interface, inputs=[user_input, state, course_selector], outputs=[user_input, chatbot])

    demo.launch()
