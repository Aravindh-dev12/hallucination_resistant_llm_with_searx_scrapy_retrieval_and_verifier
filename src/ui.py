import re

import gradio as gr

from pipeline import Pipeline

GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|how\s+are\s+you)[!.?\s]*$",
    re.IGNORECASE,
)


def build_ui():
    state = {"pipeline": None}

    def get_pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = Pipeline()
        return state["pipeline"]

    def respond(message, history):
        if not message or not message.strip():
            return ""

        clean_message = message.strip()
        if GREETING_RE.fullmatch(clean_message):
            return (
                "Hello! I’m ready. Ask me a factual question and I’ll retrieve "
                "evidence, verify the claims, cite supported sources, and abstain "
                "when reliable evidence is unavailable."
            )

        try:
            pipe = get_pipeline()
            out = pipe.answer(clean_message)
        except Exception as exc:
            state["pipeline"] = None
            return (
                "The inference service could not complete this request. "
                "It may still be warming up; please retry once.\n\n"
                f"Technical detail: {type(exc).__name__}: {exc}"
            )

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
            "Greetings respond immediately; compact models load for factual requests."
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
