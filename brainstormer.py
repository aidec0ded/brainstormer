from openai import OpenAI
import chromadb
import uuid 

# Initialize OpenAI client
client = OpenAI()
# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")



# Create or get a collection for storing the conversation
conversation_collection = chroma_client.get_or_create_collection(name="conversation_history")

def get_openai_embedding(text: str) -> list:
    """
    Returns the embedding vector for the given text using OpenAI's Embeddings API.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_response(persona_name, persona_desc, idea, context):
    """
    Generates a response from a persona, using the 'context' we retrieved from Chroma.
    """
    # Construct prompt based on persona instructions + context + user idea
    messages = [
        {
            "role": "developer",
            "content": (
                f"You are {persona_name}, who is {persona_desc}. You are participating in a dynamic brainstorming session "
                "with other experts. Engage naturally with the other participants, responding to "
                "their points while adding your unique expertise and perspective to the discussion. "
                "Ask questions, challenge assumptions constructively, and help evolve the idea. Offer new insights,"
                " reference previous points without repeating them verbatim, and move the conversation forward."
            )
        },
        {
            "role": "user", 
            "content": (
                f"{context}\n\n"
                "Please provide your next message in this discussion."
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

def store_message_in_chroma(persona_name, message):
    """
    Stores the given message in the Chroma collection, using the persona name in metadata.
    """
    embedding = get_openai_embedding(message)
    doc_id = str(uuid.uuid4())  # generate a unique ID

    conversation_collection.add(
        documents=[message],
        embeddings=[embedding],
        metadatas=[{"persona": persona_name}],
        ids=[doc_id]
    )

# def generate_summary_of_conversation(conversation_history, existing_summary):
#     full_text = ""
#     for persona, responses in conversation_history.items():
#         full_text += f"Persona: {persona}\n"
#         for resp in responses:
#             full_text += f"{resp}\n"
    
#     # Now pass full_text + existing_summary to your LLM to get a new summary
#     prompt = (
#         f"You have an existing summary of the conversation:\n{existing_summary}\n\n"
#         f"Here is the latest conversation:\n{full_text}\n\n"
#         "Please update and refine the summary in a concise way, including key points, "
#         "decisions, and new ideas. Avoid redundant detail."
#     )
    
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=500
#     )
#     return completion.choices[0].message.content.strip()

# def iterate_brainstorming(personas, idea, num_iterations=10):
#     conversation_history = {p[0]: [] for p in personas}  # Use just the name as the key
#     # Pseudocode
#     conversation_summary = ""
#     for i in range(num_iterations):
#         # Generate an updated summary of the entire conversation so far
#         conversation_summary = generate_summary_of_conversation(conversation_history, conversation_summary)
# 
#         # For each persona, combine the updated summary with the most recent round
#         for persona_name, persona_desc in personas:
#             aggregated_context = f"{conversation_summary}\n"
#             # Add just the last round's responses for immediate context
#             for other_name, responses in conversation_history.items():
#                 if responses:
#                     aggregated_context += f"\n== {other_name}'s last response ==\n{responses[-1]}\n"
#         
#             # Then generate the new response
#             next_response = generate_response(persona_name, persona_desc, idea, aggregated_context)
#             conversation_history[persona_name].append(next_response)
#     
#     return conversation_history

def run_brainstorming_with_chroma(personas, idea, total_turns_each=10, k=3):
    """
    Each persona gets 'total_turns_each' turns.
    'k' is how many relevant chunks we retrieve at each turn.
    """
    # conversation_history just for local reference or printing
    conversation_history = {p[0]: [] for p in personas}  # key: persona_name

    # We'll do total_turns = number_of_personas * total_turns_each
    num_personas = len(personas)
    total_turns = num_personas * total_turns_each

    for turn_index in range(total_turns):
        # Decide which persona goes now
        current_persona_index = turn_index % num_personas
        persona_name, persona_desc = personas[current_persona_index]

        # 1) Formulate a query to retrieve relevant context
        # We'll keep it simple: we can use the "idea" + the last message from the conversation
        # Optionally, combine the idea with the previous persona’s last message, if available.
        last_message = ""
        if conversation_history[persona_name]:
            last_message = conversation_history[persona_name][-1]
        
        # Construct a short query for retrieval
        retrieval_query = f"New round for {persona_name}. Last message from them: {last_message}. Idea: {idea}"

        # 2) Retrieve top k relevant docs
        relevant_context = retrieve_relevant_context(retrieval_query, k=k)

        # 3) Construct the final context for the model
        # We insert the relevant docs at the top or bottom as you prefer
        aggregated_context = (
            f"Relevant conversation snippets:\n{relevant_context}\n\n"
            f"Your role: {persona_name}, who is {persona_desc}.\n"
            f"Original idea: {idea}\n"
            "Build on the discussion based on the above relevant snippets.\n"
        )

        # 4) Generate a response
        next_response = generate_response(persona_name, persona_desc, idea, aggregated_context)

        # 5) Store in local history
        conversation_history[persona_name].append(next_response)

        # 6) Also store the new message in Chroma
        store_message_in_chroma(persona_name, next_response)

    return conversation_history

def retrieve_relevant_context(query_text: str, k=5):
    """
    Retrieves the top k most relevant documents from Chroma for the given query.
    We'll embed the query, then do a similarity search.
    """
    query_embedding = get_openai_embedding(query_text)
    
    results = conversation_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # results['documents'] is a list of lists (because we can have multiple queries at once)
    relevant_docs = results['documents'][0] if results and results['documents'] else []
    
    # We'll just concatenate them. You could handle them differently, e.g. bullet points, etc.
    context_text = "\n".join(relevant_docs)
    return context_text

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

    # 3. Run the new vector-DB-based brainstorming
    #    (Each persona gets 10 turns by default)
    conversation_history = run_brainstorming_with_chroma(personas, user_idea, total_turns_each=10, k=3)

    # 3. Run 10 iterations
    # conversation_history = iterate_brainstorming(personas, user_idea)


    # Print full conversation history
    print("\n=== FULL CONVERSATION HISTORY ===")
    
    # # Print introductions with full descriptions
    # for persona_name, responses in conversation_history.items():
    #     # Find matching description for this persona
    #     persona_desc = next(desc for name, desc in personas if name == persona_name)
    #     print(f"\n{persona_name} ({persona_desc}) introduction:")
    #     print(responses[0])
    
    # # Print subsequent rounds with just names
    # num_rounds = len(next(iter(conversation_history.values()))) - 1  # Subtract 1 for intro
    # for round_num in range(1, num_rounds + 1):
    #     print(f"\nRound {round_num}")
    #     for persona_name, responses in conversation_history.items():
    #         print(f"{persona_name} comments:")
    #         print(responses[round_num])

    # 4. Print the conversation or do the final synthesis
    #    (For brevity, we won’t replicate your entire “synthesize_final_output” function here.)
    # Get number of turns from first persona's responses
    num_turns = len(next(iter(conversation_history.values())))
    
    # For each turn number
    for turn in range(num_turns):
        print(f"\n=== Turn {turn + 1} ===")
        # For each persona
        for persona_name, responses in conversation_history.items():
            print(f"\n*** {persona_name} ***")
            print(responses[turn])
            print()

    # 5. Synthesize final output
    final_output = synthesize_final_output(conversation_history, personas, user_idea)
    print("\n=== FINAL OUTPUT ===")
    print(final_output)

if __name__ == "__main__":
    main()