from openai import OpenAI
import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


client = OpenAI()
tts_client = ElevenLabs()

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
                "Ask questions, challenge assumptions constructively, and help evolve the idea."
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
        temperature=0.7
    )

    return completion.choices[0].message.content.strip()

def iterate_brainstorming(personas, idea, num_iterations=10):
    conversation_history = {p[0]: [] for p in personas}  # Use just the name as the key

    for i in range(num_iterations):
        # For each persona
        for persona_name, persona_desc in personas:
            # Get recent context focusing on the last exchange
            aggregated_context = ""
            for other_name, responses in conversation_history.items():
                if responses:
                    # Only show the most recent response from each persona
                    aggregated_context += (
                        f"\n== {other_name}'s most recent comment ==\n"
                        f"{responses[-1]}\n"
                    )
            
            # Generate next response
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
                "You are a neutral assistant who can synthesize multiple inputs into a coherent, refined final concept and recommended project plan."
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's original idea:\n\n{idea}\n\n"
                f"Below is the full multi-persona conversation:\n"
                f"{chat_transcript}\n\n"
                "Please provide a finalized, refined version of the idea, and a recommended project plan with clear next steps."
            )
        }
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5000,
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()

def text_to_speech_file(text: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = tts_client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    # uncomment the line below to play the audio back
    # play(response)
    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")
    # Return the path of the saved audio file
    return save_file_path

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

    # 5. Save audio file
    text_to_speech_file(final_output)

if __name__ == "__main__":
    main()