from html import escape
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from retrieving_and_answering.llm_answer import answer_question

load_dotenv(override=True)

WARM_CSS = """
:root {
    --paper: #f4f3ee;
    --surface: #fbfaf7;
    --ink: #223c34;
    --text: #1B2921;
    --muted: #5b6a60;
    --line: #e1e5dc;
    --sage: #edf2eb;
    --copper: #bb7951;
}
html, body, .gradio-container {
    min-height: 100vh !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 0 0 32px !important;
}
#topbar, #hero, #main-workspace { max-width: 1480px; margin-left: auto; margin-right: auto; }
#topbar { padding: 18px 36px 14px; border-bottom: 1px solid var(--line); }
.topbar-inner { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.brand { font-size: 21px; font-weight: 750; letter-spacing: -.04em; color: var(--ink); }
.brand-mark { display: inline-grid; place-items: center; width: 31px; height: 31px; margin-right: 10px; border-radius: 10px; background: var(--ink); color: white; font-size: 17px; }
.knowledge-pill, .grounded-pill { display: inline-flex; align-items: center; gap: 8px; background: var(--sage); border: 1px solid var(--line); color: var(--text); border-radius: 999px; padding: 8px 13px; font-size: 12px; font-weight: 650; }
.status-dot { width: 7px; height: 7px; background: #6e9a78; border-radius: 50%; }
#hero { padding: 38px 36px 28px; }
.hero-eyebrow, .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: 11px; font-weight: 750; color: var(--muted); }
.hero-eyebrow span { color: var(--copper); }
.hero-title { margin: 10px 0 8px; font-family: Georgia, serif; font-size: clamp(32px, 4vw, 52px); font-weight: 500; line-height: 1.1; letter-spacing: -.055em; color: var(--ink); }
.hero-title em { color: var(--copper); font-weight: 400; }
.hero-bottom { display: flex; align-items: end; justify-content: space-between; gap: 20px; }
.hero-copy { max-width: 600px; margin: 0; font-size: 15px; line-height: 1.65; color: var(--text); }
#main-workspace { padding: 0 36px; gap: 22px !important; align-items: stretch; min-width: 0; box-sizing: border-box; }
#conversation-column, #evidence-column { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 14px 38px rgba(35, 59, 48, .045); overflow: hidden; min-width: 0; }
#conversation-column { padding: 0 0 17px !important; }
#evidence-column { padding: 22px !important; }
.chat-card-header { display: flex; align-items: center; gap: 12px; padding: 18px 24px; border-bottom: 1px solid var(--line); }
.card-icon { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 12px; background: var(--sage); color: var(--ink); font-size: 18px; }
.card-title { color: var(--ink); font-size: 14px; font-weight: 750; }
.card-subtitle { margin-top: 3px; color: var(--muted); font-size: 11px; }
#conversation { border: 0 !important; background: var(--surface) !important; border-radius: 0 !important; min-width: 0 !important; overflow-x: hidden !important; }
#conversation .bubble-wrap { background: var(--surface) !important; color: var(--text) !important; overflow-x: hidden !important; }
#conversation .message-wrap, #conversation .message-row, #conversation .message-content, #conversation .md { max-width: 100% !important; box-sizing: border-box; }
#conversation .bot-row { width: 100% !important; }
#conversation .message.user { background: #eaf0e9 !important; color: var(--ink) !important; border: 0 !important; box-shadow: none !important; border-radius: 17px 17px 5px 17px !important; }
#conversation .message.user *, #conversation .message.bot * { color: inherit !important; }
#conversation .message.bot { background: transparent !important; color: #142019 !important; border: 0 !important; box-shadow: none !important; }
#conversation .placeholder, #conversation .placeholder p { color: var(--muted) !important; }
#conversation .message.bot a { color: var(--copper) !important; }
#conversation .message-buttons button { color: var(--ink) !important; visibility: visible !important; opacity: 1 !important; }
#conversation-column .form { background: var(--surface) !important; border: 0 !important; }
#question-input { margin: 8px 18px 0 !important; width: calc(100% - 36px) !important; min-width: 0 !important; max-width: calc(100% - 36px) !important; box-sizing: border-box !important; }
#question-input textarea { background: #f6f7f3 !important; color: var(--ink) !important; border-color: var(--line) !important; border-radius: 12px !important; min-height: 56px !important; padding: 15px 10px !important; }
#question-input textarea::placeholder { color: var(--muted) !important; }
#question-input:focus-within { border-color: #93aa98 !important; }
#question-input button { background: var(--ink) !important; color: white !important; border-radius: 10px !important; }
.evidence-heading { margin: 11px 0 8px; color: var(--ink); font-family: Georgia, serif; font-size: 25px; letter-spacing: -.04em; }
.evidence-intro { margin: 0 0 19px; color: var(--muted); font-size: 12px; line-height: 1.5; }
.source-count { display: inline-block; margin-left: 8px; padding: 5px 8px; border-radius: 999px; background: var(--sage); color: var(--text); font-family: sans-serif; font-size: 10px; font-weight: 750; letter-spacing: 0; vertical-align: middle; }
.source-list { display: grid; gap: 10px; }
.source-card { display: flex; align-items: center; gap: 12px; padding: 15px; border: 1px solid var(--line); border-radius: 13px; background: #f9faf7; }
.source-icon { display: inline-grid; place-items: center; flex: 0 0 33px; height: 33px; border-radius: 10px; background: #e8eee7; color: var(--ink); font-size: 17px; }
.source-name { overflow-wrap: anywhere; color: var(--ink); font-size: 13px; font-weight: 650; }
.empty-sources { padding: 17px; border: 1px dashed var(--line); border-radius: 12px; color: var(--muted); font-size: 13px; line-height: 1.5; }
.evidence-note { margin-top: 20px; padding: 12px; border-radius: 10px; background: var(--sage); color: var(--text); font-size: 11px; line-height: 1.5; }
@media (max-width: 800px) {
    #topbar { padding: 16px; }
    #hero { padding: 30px 16px 23px; }
    #main-workspace { padding: 0 16px; flex-direction: column !important; }
    #conversation-column, #evidence-column { width: 100% !important; max-width: 100% !important; min-width: 0 !important; flex: 0 0 auto !important; }
    .hero-bottom { display: block; }
    .grounded-pill { margin-top: 16px; }
}
"""


def format_sources(context):
    names = []
    for doc in context:
        source = doc.metadata.get("source") or doc.metadata.get("file_name")
        if source:
            name = Path(str(source).replace("\\", "/")).name
            if name not in names:
                names.append(name)

    cards = "".join(
        '<div class="source-card"><span class="source-icon">▤</span>'
        f'<span class="source-name">{escape(name)}</span></div>'
        for name in names
    ) or '<div class="empty-sources">Sources for the latest answer will appear here.</div>'

    return (
        '<div class="eyebrow">— Evidence</div>'
        '<h2 class="evidence-heading">Where this comes from'
        f'<span class="source-count">{len(names)} SOURCES</span></h2>'
        '<p class="evidence-intro">Documents retrieved for the latest answer.</p>'
        f'<div class="source-list">{cards}</div>'
        '<div class="evidence-note">For decisions that affect you, check the latest university policy.</div>'
    )


def chat(history):
    if not history or history[-1]["role"] != "user":
        return history, gr.skip()

    content = history[-1]["content"]
    last_message = (
        content if isinstance(content, str)
        else " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    )
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_sources(context)


def main():
    def put_message_in_chatbot(message, history):
        history = history or []
        if not message or not message.strip():
            return "", history
        return "", history + [{"role": "user", "content": message.strip()}]

    theme = gr.themes.Soft(
        font=["Inter", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f4f3ee",
        body_background_fill_dark="#f4f3ee",
        body_text_color="#223c34",
        body_text_color_dark="#223c34",
        background_fill_primary="#fbfaf7",
        background_fill_primary_dark="#fbfaf7",
        block_background_fill="#fbfaf7",
        block_background_fill_dark="#fbfaf7",
        color_accent="#bb7951",
        color_accent_soft="#edf2eb",
        color_accent_soft_dark="#edf2eb",
    )

    with gr.Blocks(
        title="UniQuery — Campus guide",
        fill_width=True,
    ) as ui:
        gr.HTML(
            '<div class="topbar-inner">'
            '<div class="brand"><span class="brand-mark">✦</span>UniQuery</div>'
            '<div class="knowledge-pill"><span class="status-dot"></span>University knowledge base</div>'
            '</div>',
            elem_id="topbar",
        )
        gr.HTML(
            '<div class="hero-eyebrow">Student guide &nbsp;/&nbsp; <span>Ask with confidence</span></div>'
            '<h1 class="hero-title">Good questions deserve <em>clear answers.</em></h1>'
            '<div class="hero-bottom">'
            '<p class="hero-copy">A thoughtful guide to university information, grounded in documents you can check for yourself.</p>'
            '<div class="grounded-pill">◇ &nbsp; Answers grounded in university documents</div>'
            '</div>',
            elem_id="hero",
        )

        with gr.Row(equal_height=True, elem_id="main-workspace"):
            with gr.Column(scale=7, min_width=350, elem_id="conversation-column"):
                gr.HTML(
                    '<div class="chat-card-header">'
                    '<span class="card-icon">✦</span>'
                    '<div><div class="card-title">Campus guide</div>'
                    '<div class="card-subtitle">Answers from university documents</div></div>'
                    '</div>'
                )
                chatbot = gr.Chatbot(
                    show_label=False,
                    container=False,
                    height=500,
                    layout="bubble",
                    buttons=["copy", "share"],
                    placeholder="Ask a question about university policies or services.",
                    elem_id="conversation",
                )
                message = gr.Textbox(
                    placeholder="Ask a question about your university...",
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    submit_btn="Ask",
                    elem_id="question-input",
                )

            with gr.Column(scale=4, min_width=300, elem_id="evidence-column"):
                sources = gr.HTML(
                    value=format_sources([]),
                    elem_id="source-panel",
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, sources])

    ui.launch(
        inbrowser=True,
        theme=theme,
        css=WARM_CSS,
    )


if __name__ == "__main__":
    main()
