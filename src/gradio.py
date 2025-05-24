import gradio as gr
from src.retriever import RAG
from src.config import Config

rag = RAG()

def chat_interface(user_input, history, selected_course):
    response = rag.get_response(user_input, history, selected_course)
    history.append((user_input, response))
    
    return "", history

def clear_history():
    return [], []

def launch_ui():
    with gr.Blocks(title="RAG QA Chatbot") as demo:
        gr.Markdown("# 📚 UM Faculty Handbook QA Bot\nAsk questions about your faculty handbook for the 2024/2025 academic session.")

        with gr.Row():
            course_selector = gr.Dropdown(
                choices=list(Config.COURSE_TO_FILE_MAP.keys()),
                label="Select Your Course/Faculty",
                interactive=True,
        )
            
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Hi! How can I assist you today?", show_label=False)

        state = gr.State([])  

        user_input.submit(chat_interface, inputs=[user_input, state, course_selector], outputs=[user_input, chatbot])
        chatbot.clear(clear_history, outputs=[state, chatbot])  

    # demo.launch(share=True)
    demo.launch()
