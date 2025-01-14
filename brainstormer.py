from openai import OpenAI

client = OpenAI()

def generate_response(persona_name, persona_desc, idea, context):
    """Generates a response from a persona given an idea and context."""
    # Construct prompt based on persona instructions + context + user idea
    messages = [
        {
            "role": "developer",
            "content": (
                f"You are {persona_name}, who is {persona_desc}. You are participating in a dynamic brainstorming session "
                "with other experts. Engage naturally with the other participants, responding to "
                "their points while adding your unique expertise and perspective to the discussion. "
                "Ask questions, challenge assumptions constructively, and help evolve the idea. Refrain from "
                "repeating yourself and engaging in circular conversations. Always try to inject new ideas or "
                "make concrete recommendations for existing ones."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Original idea being discussed:\n{idea}\n\n"
                f"Recent conversation:\n{context}\n\n"
                "Based on the conversation above, continue the discussion. Respond to specific points "
                "raised by other participants while advancing the conversation with your expertise. "
                "If this is your first response, provide your initial reaction to the idea."
            )
        }
    ]

    # Call your LLM of choice
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000,
        temperature=0.8
    )

    return completion.choices[0].message.content.strip()

def generate_summary_of_conversation(conversation_history, existing_summary):
    full_text = ""
    for persona, responses in conversation_history.items():
        full_text += f"Persona: {persona}\n"
        for resp in responses:
            full_text += f"{resp}\n"
    
    # Now pass full_text + existing_summary to your LLM to get a new summary
    prompt = (
        f"You have an existing summary of the conversation:\n{existing_summary}\n\n"
        f"Here is the latest conversation:\n{full_text}\n\n"
        "Please update and refine the summary in a concise way, including key points, "
        "decisions, and new ideas. Avoid redundant detail."
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return completion.choices[0].message.content.strip()

def iterate_brainstorming(personas, idea, num_iterations=10):
    conversation_history = {p[0]: [] for p in personas}  # Use just the name as the key
    # Pseudocode
    conversation_summary = ""
    for i in range(num_iterations):
        # Generate an updated summary of the entire conversation so far
        conversation_summary = generate_summary_of_conversation(conversation_history, conversation_summary)

        # For each persona, combine the updated summary with the most recent round
        for persona_name, persona_desc in personas:
            aggregated_context = f"{conversation_summary}\n"
            # Add just the last round’s responses for immediate context
            for other_name, responses in conversation_history.items():
                if responses:
                    aggregated_context += f"\n== {other_name}'s last response ==\n{responses[-1]}\n"
        
            # Then generate the new response
            next_response = generate_response(persona_name, persona_desc, idea, aggregated_context)
            conversation_history[persona_name].append(next_response)
    
    return conversation_history

def synthesize_final_output(conversation_history, personas, idea):
    # Combine all conversation_history entries into a single summary
    chat_transcript = ""
    for persona_name, responses in conversation_history.items():
        # Find matching description for this persona
        persona_desc = next(desc for name, desc in personas if name == persona_name)
        chat_transcript += f"\n--- {persona_name} ({persona_desc}) ---\n"
        for i, r in enumerate(responses, start=1):
            chat_transcript += f"Round {i}:\n{r}\n"
    
    messages = [
        {
            "role": "developer",
            "content": (
                "You are a world-class management consultant. You specialize in detailed, "
                "executive-level proposals with robust data and analysis."
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's original idea:\n\n{idea}\n\n"
                f"Below is the full multi-persona conversation:\n"
                f"{chat_transcript}\n\n"
                # "Please provide a finalized, refined version of the idea, and a recommended project plan with clear next steps."
                "Please provide a comprehensive proposal in the following structure:\n\n"
                "1. Executive Summary\n2. Situation Analysis\n3. Proposed Solution\n4. "
                "Implementation Roadmap with Timelines\n5. Financials/ROI\n6. Risk Mitigation\n\n"
                "Use the conversation transcript and the user’s original idea as context."
                "Whenever it supports the proposal and brings value, include bullet points, tables, and other visual elements."
                "Identify all of the potential features mentioned in the conversation transcript and categorize them and list them as bullet points."
            )
        }
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5000,
        temperature=0.6
    )
    return completion.choices[0].message.content.strip()

def main():
    # 1. User inputs idea
    user_idea = input("What's your idea?\n> ")

    # 2. User defines personas with separate name and description
    print("Please define three personas. For each, provide their name and a description of their qualities.")
    personas = []
    for i in range(3):
        print(f"\nPersona {i+1}:")
        name = input("Name (e.g. 'Steve Jobs'):\n> ")
        desc = input("Description (e.g. 'perfectionist, design-oriented, focused on user experience'):\n> ")
        personas.append((name, desc))

    # 3. Run 10 iterations
    conversation_history = iterate_brainstorming(personas, user_idea)

    # Print full conversation history
    print("\n=== FULL CONVERSATION HISTORY ===")
    
    # Print introductions with full descriptions
    for persona_name, responses in conversation_history.items():
        # Find matching description for this persona
        persona_desc = next(desc for name, desc in personas if name == persona_name)
        print(f"\n{persona_name} ({persona_desc}) introduction:")
        print(responses[0])
    
    # Print subsequent rounds with just names
    num_rounds = len(next(iter(conversation_history.values()))) - 1  # Subtract 1 for intro
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}")
        for persona_name, responses in conversation_history.items():
            print(f"{persona_name} comments:")
            print(responses[round_num])

    # 4. Synthesize final output
    final_output = synthesize_final_output(conversation_history, personas, user_idea)
    print("\n=== FINAL OUTPUT ===")
    print(final_output)

if __name__ == "__main__":
    main()