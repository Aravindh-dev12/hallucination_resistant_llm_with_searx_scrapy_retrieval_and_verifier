import gradio as gr

from pipeline import Pipeline


def build_ui():
    state = {"pipeline": None, "error": None}

    def get_pipeline():
        if state["pipeline"] is None and state["error"] is None:
            try:
                state["pipeline"] = Pipeline()
            except Exception as exc:
                state["error"] = str(exc)
        return state["pipeline"]

    def respond(message, history):
        if not message or not message.strip():
            return ""

        pipe = get_pipeline()
        if pipe is None:
            raise gr.Error(
                "The verification pipeline could not start: "
                + (state["error"] or "unknown initialization error")
            )

        out = pipe.answer(message.strip())
        verification = out["verification"]
        display = (
            out["answer"]
            + f"\n\n**Status:** {out['status']}"
            + f"\n\n**Claim faithfulness:** {verification['faithfulness']:.3f}"
            + f"\n\n**Supported:** {verification['supported_claims']}/"
            + f"{verification['total_claims']}"
        )
        if out["citations"]:
            formatted = []
            for citation in out["citations"]:
                source = citation.get("source") or f"Src#{citation['idx']}"
                formatted.append(
                    f"Claim {citation['claim_id']}: {source} "
                    f"(entailment {citation['entailment_score']:.3f})"
                )
            display += "\n\n**Verified citations:**\n- " + "\n- ".join(formatted)
        return display

    with gr.Blocks(title="Hallucination-Resistant LLM") as demo:
        gr.Markdown(
            "# Evidence-Grounded LLM\n"
            "Hybrid retrieval, claim verification, citations, and safe abstention. "
            "The models load on the first request."
        )
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(type="messages"),
            type="messages",
        )
    return demo


def launch_ui():
    build_ui().queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
    )


if __name__ == "__main__":
    launch_ui()
