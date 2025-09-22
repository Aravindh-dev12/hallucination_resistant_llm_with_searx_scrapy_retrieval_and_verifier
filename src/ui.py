import gradio as gr
from pipeline import Pipeline

def launch_ui():
    pipe = Pipeline()
    def respond(chat_history, user_input):
        if not user_input:
            return chat_history, ''
        chat_history = chat_history or []
        chat_history.append(('User', user_input))
        out = pipe.answer(user_input)
        display = out['answer'] + f"\n\n[Verifier: {out['verifier_score']:.3f}]" + ("\n[FLAGGED]" if out['flagged'] else "")
        if out['citations']:
            display += '\n\nCitations: ' + ', '.join([f"Src#{c['idx']}" for c in out['citations']])
        chat_history.append(('Assistant', display))
        return chat_history, ''
    with gr.Blocks() as demo:
        gr.Markdown('# Hallucination-Resistant LLM — Searx+Scrapy UI')
        chat = gr.Chatbot()
        txt = gr.Textbox(placeholder='Ask a question...')
        send = gr.Button('Send')
        send.click(respond, inputs=[chat, txt], outputs=[chat, txt])
    demo.launch(share=False, inbrowser=True)

if __name__ == '__main__':
    launch_ui()
