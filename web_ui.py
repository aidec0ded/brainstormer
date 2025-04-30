import gradio as gr
from personas import PERSONA_LIBRARY
from app import (
    run_brainstorming_with_reasoning,
    synthesize_final_output,
    create_new_conversation_collection,
    initialize_persona_collection,
    manager_agent_decide_personas
)

# ✅ Display name → real name mapping
persona_display_map = {
    f"{p['name']} – {p['short_bio']}": p["name"] for p in PERSONA_LIBRARY
}
persona_display_list = list(persona_display_map.keys())

# ✅ Search bar dynamic filtering
def filter_personas(query):
    query = query.lower()
    return [
        name for name in persona_display_list
        if query in name.lower()
    ]

# ✅ Save final output to file
def download_final_output(text):
    file_path = "brainstorm_proposal.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

# ✅ Main logic with streaming output
def run_full_session(user_idea, selection_mode, manual_displayed_names, semantic_description):
    output = "🔄 Setting up environment...\n"
    yield output
    create_new_conversation_collection()
    initialize_persona_collection()

    if not user_idea.strip():
        yield output + "❌ Please enter an idea."
        return

    selected_personas = []

    if selection_mode == "manual":
        if not manual_displayed_names:
            yield output + "❌ Please select at least one persona."
            return
        selected_personas = [persona_display_map[d] for d in manual_displayed_names]

    elif selection_mode == "semantic":
        if not semantic_description.strip():
            yield output + "❌ Please enter a description for semantic search."
            return
        selected_personas = [
            p["name"] for p in PERSONA_LIBRARY
            if any(word.lower() in p["desc"].lower() for word in semantic_description.split())
        ][:5]
        if not selected_personas:
            yield output + "❌ No personas matched. Try again."
            return

    else:
        output += "🧠 Letting the manager agent select personas...\n"
        yield output
        selected_personas = manager_agent_decide_personas(user_idea)

    if not selected_personas:
        yield output + "❌ No personas were selected."
        return

    output += f"✅ Selected personas: {', '.join(selected_personas)}\n"
    output += "💬 Brainstorming session in progress...\n"
    yield output

    history = run_brainstorming_with_reasoning(
        persona_names=selected_personas,
        idea=user_idea,
        total_turns_each=2
    )

    output += "🧠 Brainstorm Transcript:\n"
    total_turns = len(history[selected_personas[0]])
    for turn_index in range(total_turns * len(selected_personas)):
        persona = selected_personas[turn_index % len(selected_personas)]
        round_number = turn_index // len(selected_personas)
        if round_number < len(history[persona]):
            output += f"\nTurn {turn_index + 1} – {persona}: {history[persona][round_number]}\n"
            yield output

    output += "\n📝 Generating final proposal...\n"
    yield output
    final_output = synthesize_final_output(history, selected_personas, user_idea)
    output += "\n✅ Final Proposal:\n\n" + final_output
    yield output

# 🧱 Gradio UI
with gr.Blocks(title="Brainstormer", css="""
#persona-scroll-box > div {
    max-height: 250px !important;
    overflow-y: auto;
    background-color: transparent;
    padding-right: 6px;
    border: none;
}

input[type='checkbox']:checked + label {
    background-color: #e0f7fa !important;
    border-radius: 6px;
    font-weight: bold;
}

label[title] {
    cursor: help;
}
""") as demo:

    gr.Markdown("## 🤖 Brainstormer")
    gr.Markdown("Generate a consulting-style proposal using multiple AI personas. See how they think before summarizing!")

    with gr.Row():
        with gr.Column(scale=1):
            user_idea = gr.Textbox(
                label="Your Idea",
                lines=3,
                placeholder="e.g. A platform to help students find roommates"
            )

            selection_mode = gr.Radio(
                ["auto", "semantic", "manual"],
                value="auto",
                label="Persona Selection Mode"
            )

            persona_search = gr.Textbox(
                label="Search Personas",
                visible=False,
                placeholder="Type to filter personas (Manual Mode)"
            )

            with gr.Column(visible=False, elem_id="persona-scroll-box") as manual_scroll_wrap:
                manual_selection = gr.CheckboxGroup(
                    choices=persona_display_list,
                    label="Select Personas (Manual Mode)"
                )

            semantic_description = gr.Textbox(
                label="Describe Persona Type (Semantic Search)",
                placeholder="e.g. UX expert with mobile app experience",
                visible=False
            )

            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            final_output = gr.Markdown(
                label="Progress, Brainstorming Transcript, and Final Proposal"
            )
            download_button = gr.Button("📥 Download Final Proposal")
            download_file = gr.File(label="Download Link")
            final_output_copy = gr.Textbox(
                visible=False, show_copy_button=True
            )

    def toggle_inputs(mode):
        return (
            gr.update(visible=(mode == "manual")),
            gr.update(visible=(mode == "manual")),
            gr.update(visible=(mode == "semantic"))
        )

    selection_mode.change(
        toggle_inputs,
        inputs=selection_mode,
        outputs=[manual_scroll_wrap, persona_search, semantic_description]
    )

    persona_search.change(
        fn=filter_personas,
        inputs=persona_search,
        outputs=manual_selection
    )

    submit_btn.click(
        fn=run_full_session,
        inputs=[user_idea, selection_mode, manual_selection, semantic_description],
        outputs=final_output
    )

    download_button.click(
        fn=download_final_output,
        inputs=final_output,
        outputs=download_file
    )

    clear_btn.click(
        fn=lambda: ("", "auto", [], "", "", None),
        inputs=[],
        outputs=[user_idea, selection_mode, manual_selection, semantic_description, final_output, download_file]
    )

    demo.launch(share=True, inline=False, height=900)

# main previous 