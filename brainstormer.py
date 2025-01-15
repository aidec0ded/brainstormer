from openai import OpenAI
import chromadb
import uuid 

# Initialize OpenAI client
client = OpenAI()
# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Define personas
PERSONA_LIBRARY = [
    {
        "name": "Rebecca",
        "desc": (
            "Rebecca is a visionary entrepreneur and trendspotter. She excels at ideation, brand strategy, and finding "
            "untapped opportunities in AI-driven consumer products. Energetic, bold, and optimistic, Rebecca always "
            "pushes for the big-picture vision and aims to disrupt markets with daring ideas."
        )
    },
    {
        "name": "Leo",
        "desc": (
            "Leo is an ethical AI guardian who prioritizes fairness, transparency, and data protection. With a background "
            "in law and policy, he constantly evaluates potential social impacts of new technology. Calm and introspective, "
            "Leo challenges the group to address moral and regulatory issues."
        )
    },
    {
        "name": "Joy",
        "desc": (
            "Joy is an artistic technologist with a passion for creative coding, AR/VR experiences, and cutting-edge design. "
            "She merges art and engineering seamlessly. Joy is playful, curious, and likes to experiment with new media "
            "to surprise and delight end users."
        )
    },
    {
        "name": "Amir",
        "desc": (
            "Amir is a data scientist and machine learning researcher who loves algorithms, analytics, and "
            "statistical rigor. With a methodical, evidence-based mindset, he always wants to test assumptions, "
            "run simulations, and optimize models for performance."
        )
    },
    {
        "name": "Sophie",
        "desc": (
            "Sophie is a market strategist and product marketer who focuses on product-market fit, go-to-market strategies, "
            "and competitive landscapes. She’s driven by real-world user needs, ROI, and clear messaging, ensuring ideas "
            "align to business goals and resonate with customers."
        )
    },
    {
        "name": "Lucas",
        "desc": (
            "Lucas is a UI/UX perfectionist, obsessed with simplicity and a top-notch user journey. With a user-first "
            "mentality, he advocates for minimal friction, delightful interactions, and thoughtful accessibility in every design."
        )
    },
    {
        "name": "Xavier",
        "desc": (
            "Xavier is security-conscious and risk-averse, always anticipating data breaches, regulatory fines, or "
            "unknown vulnerabilities. He enforces strict security protocols and weighs worst-case scenarios. "
            "While sometimes overly cautious, he balances innovation with safety."
        )
    },
    {
        "name": "Natasha",
        "desc": (
            "Natasha is an AI philosopher and futurist, versed in cognitive science, ethics, and existential risks. "
            "She’s fascinated by how AI shapes consciousness, identity, and society. Natasha’s sweeping insights drive "
            "the team to think deeply about the human-AI relationship."
        )
    },
    {
        "name": "Hiro",
        "desc": (
            "Hiro is a scalability-focused architect who specializes in cloud infrastructure, microservices, and "
            "distributed systems. He’s pragmatic and performance-driven, designing solutions that handle massive "
            "traffic, data, and concurrency without breaking a sweat."
        )
    },
    {
        "name": "Anya",
        "desc": (
            "Anya is an expert in immersive experiences and gamification, bridging VR/AR with everyday apps. She finds "
            "ways to keep users engaged through playful mechanics and interactive narratives, believing that fun, "
            "immersive design can enhance productivity and creativity."
        )
    }
]

# Create or get a collection for storing the conversation
conversation_collection = chroma_client.get_or_create_collection(name="conversation_history")

# Create a dedicated persona collection
persona_collection = chroma_client.get_or_create_collection(name="persona_library")

def get_openai_embedding(text: str) -> list:
    """
    Returns the embedding vector for the given text using OpenAI's Embeddings API.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_response_for_persona(persona_name, idea, context):
    """
    Dynamically retrieves the persona's 'essence' from Chroma and injects it into the system or developer message.
    """
    persona_desc = retrieve_persona_by_name(persona_name)

    messages = [
        {
            "role": "developer",
            "content": (
                f"You are {persona_name}. Below is your personality description or 'essence':\n\n"
                f"{persona_desc}\n\n"
                "Always leverage this mindset, worldview, and expertise. "
                "You are participating in a dynamic brainstorming session with other experts. "
                "Engage naturally with the other participants, responding to their points while adding your unique "
                "expertise and perspective to the discussion. "
                "Ask questions, challenge assumptions constructively, and help evolve the idea. Offer new insights,"
                " reference previous points without repeating them verbatim, and move the conversation forward."
                "Contribute to the discussion in a way that aligns with your persona's style."
            )
        },
        {
            "role": "user", 
            "content": (
                f"Original idea: {idea}\n\n"
                f"Relevant conversation context: \n{context}\n\n"
                "Please provide your next message in this brainstorming session."
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

def store_personas_in_chroma(personas):
    """
    For each persona, embed their description and store in the 'persona_library' collection.
    """
    for p in personas:
        persona_name = p["name"]
        persona_desc = p["desc"]

        emb = get_openai_embedding(persona_desc)  # e.g., "text-embedding-ada-002"
        doc_id = f"persona-{persona_name.lower().replace(' ', '-')}"  # or UUID
        
        persona_collection.add(
            documents=[persona_desc],
            embeddings=[emb],
            metadatas=[{"persona_name": persona_name}],
            ids=[doc_id]
        )

def retrieve_persona_by_name(persona_name: str) -> str:
    """
    Fetches the persona description from the persona_library collection, using metadata filtering.
    """
    # A simple approach is to iterate over the collection or filter by 'persona_name'
    # Chroma doesn't have a direct "where" filter on metadata just yet, so we could do a similarity
    # search with the name, or keep a local dictionary for direct name-based lookup.
    
    # Let's do a naive similarity search:
    emb = get_openai_embedding(persona_name)
    results = persona_collection.query(query_embeddings=[emb], n_results=1)
    # Return the top doc
    if results and results['documents']:
        return results['documents'][0][0]  # first doc of first query
    return ""

def run_brainstorming_with_personas(persona_names, idea, total_turns_each=10, k=3):
    """
    'persona_names' is a list of persona names from our persona library in Chroma.
    Each persona gets 'total_turns_each' opportunities to speak.
    """
    conversation_history = {name: [] for name in persona_names}
    num_personas = len(persona_names)
    total_turns = num_personas * total_turns_each

    for turn_index in range(total_turns):
        current_persona_index = turn_index % num_personas
        persona_name = persona_names[current_persona_index]
        
        # Formulate a retrieval query
        last_message = conversation_history[persona_name][-1] if conversation_history[persona_name] else ""
        retrieval_query = f"New turn for {persona_name}. Last message from them: {last_message}. Idea: {idea}"
        
        # Retrieve top k relevant docs from the conversation collection
        relevant_context = retrieve_relevant_context(retrieval_query, k=k)
        
        # Generate persona’s response with their “essence”
        next_response = generate_response_for_persona(persona_name, idea, relevant_context)
        
        # Store in local history + vector DB
        conversation_history[persona_name].append(next_response)
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

def synthesize_final_output(conversation_history, persona_names, idea):
    # Combine all conversation_history entries into a single summary
    chat_transcript = ""
    
    # Get the number of turns from the first persona's history
    total_turns = len(conversation_history[persona_names[0]])
    num_personas = len(persona_names)
    
    # For each turn, find which persona was speaking based on the turn number
    for turn_index in range(total_turns * num_personas):
        current_persona_index = turn_index % num_personas
        persona_name = persona_names[current_persona_index]
        round_number = turn_index // num_personas
        
        # Only add to transcript if this persona still has turns left
        if round_number < len(conversation_history[persona_name]):
            # Find matching description for this persona from PERSONA_LIBRARY
            persona_desc = next(p["desc"] for p in PERSONA_LIBRARY if p["name"] == persona_name)
            chat_transcript += f"\n--- Turn {turn_index + 1}: {persona_name} ({persona_desc}) ---\n"
            chat_transcript += f"{conversation_history[persona_name][round_number]}\n"

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
    # 1) First, store the 10 persona definitions in Chroma.
    #    (Only do this once per application run or when you add a new persona)
    store_personas_in_chroma(PERSONA_LIBRARY)
    
    # 2. Ask the user for the idea.
    user_idea = input("What's your idea?\n> ")

    # 3) Let the user pick from the persona library
    print("Available personas in library:")
    for i, p in enumerate(PERSONA_LIBRARY, start=1):
        print(f"{i}. {p['name']}")
    
    selected_indices = input("Pick 3 personas by entering indices separated by commas (e.g., 1,3,5):\n> ")
    selected_indices = [int(x.strip()) for x in selected_indices.split(",")]

    # 4) Extract the persona names
    selected_persona_names = [PERSONA_LIBRARY[i-1]["name"] for i in selected_indices]

    # 5) Run the brainstorming loop
    conversation_history = run_brainstorming_with_personas(
        persona_names=selected_persona_names,
        idea=user_idea,
        total_turns_each=10,
        k=3
    )

    # 6) Print out the final conversation in round-robin order
    # Get the number of turns from the first persona's history
    total_turns = len(conversation_history[selected_persona_names[0]])
    num_personas = len(selected_persona_names)
    
    # Print full conversation history
    print("\n=== FULL CONVERSATION HISTORY ===")

    # For each turn, find which persona was speaking based on the turn number
    for turn_index in range(total_turns * num_personas):
        current_persona_index = turn_index % num_personas
        persona_name = selected_persona_names[current_persona_index]
        round_number = turn_index // num_personas
        
        # Only print if this persona still has turns left
        if round_number < len(conversation_history[persona_name]):
            print(f"\n{persona_name}, Turn {round_number + 1}:")
            print(f"{conversation_history[persona_name][round_number]}\n")

    # 7. Synthesize final output
    final_output = synthesize_final_output(conversation_history, selected_persona_names, user_idea)
    print("\n=== FINAL OUTPUT ===")
    print(final_output)

if __name__ == "__main__":
    main()