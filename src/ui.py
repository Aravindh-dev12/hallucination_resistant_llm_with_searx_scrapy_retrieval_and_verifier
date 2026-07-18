import gradio as gr

from pipeline import Pipeline


def launch_ui():
    pipe = Pipeline()

    def respond(chat_history, user_input):
        if not user_input:
            return chat_history, ""
        chat_history = chat_history or []
        chat_history.append(("User", user_input))
        out = pipe.answer(user_input)
        verification = out["verification"]
        display = (
            out["answer"]
            + f"\n\n[Status: {out['status']}]"
            + f"\n[Claim faithfulness: {verification['faithfulness']:.3f}]"
            + f"\n[Supported: {verification['supported_claims']}/"
            + f"{verification['total_claims']}]"
        )
        if out["citations"]:
            formatted = []
            for citation in out["citations"]:
                source = citation.get("source") or f"Src#{citation['idx']}"
                formatted.append(
                    f"Claim {citation['claim_id']}: {source} "
                    f"(entailment {citation['entailment_score']:.3f})"
                )
            display += "\n\nVerified citations:\n- " + "\n- ".join(formatted)
        chat_history.append(("Assistant", display))
        return chat_history, ""

    with gr.Blocks() as demo:
        gr.Markdown("# Evidence-Grounded LLM — Hybrid Retrieval + Claim Verification")
        chat = gr.Chatbot()
        txt = gr.Textbox(placeholder="Ask a question...")
        send = gr.Button("Send")
        send.click(respond, inputs=[chat, txt], outputs=[chat, txt])
    demo.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    launch_ui()
