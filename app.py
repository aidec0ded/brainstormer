from openai import OpenAI
import chromadb
import uuid 
import datetime
import logging
from personas import PERSONA_LIBRARY

# Initialize OpenAI client
client = OpenAI()
# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

SESSION_COLLECTION = None
SESSION_ID = None

# Create a new or existing archive collection:
archive_collection = chroma_client.get_or_create_collection(name="all_session_archives")

def is_persona_collection_current():
    """
    Checks if the persona collection exists and is up to date.
    Returns True if collection exists and matches current PERSONA_LIBRARY.
    """
    try:
        # Try to get the collection
        collection = chroma_client.get_collection(name="persona_library")
        
        # Get all stored personas
        results = collection.get()
        stored_personas = {
            meta["persona_name"]: doc 
            for meta, doc in zip(results["metadatas"], results["documents"])
        }
        
        # Check if all current personas exist and match
        for persona in PERSONA_LIBRARY:
            name = persona["name"]
            if name not in stored_personas:
                return False
            # Check if description matches
            if stored_personas[name] != persona["desc"]:
                return False
                
        return True
    except:
        return False

def create_new_conversation_collection():
    global SESSION_COLLECTION, SESSION_ID
    # Use timestamp or short UUID for uniqueness
    unique_id = str(uuid.uuid4())[:8]
    SESSION_ID = f"session_{unique_id}"
    
    SESSION_COLLECTION = chroma_client.get_or_create_collection(name=SESSION_ID)
    print(f"Created new conversation collection: {SESSION_ID}")

def initialize_persona_collection():
    """Initialize or update collections as needed."""
    global persona_collection

    # Temporarily disable ChromaDB logging
    chromadb_logger = logging.getLogger('chromadb')
    original_level = chromadb_logger.level
    chromadb_logger.setLevel(logging.ERROR)  # Only show errors
    
    # Check persona collection
    if not is_persona_collection_current():
        print("Initializing persona collection (this may take a moment)...")
        try:
            chroma_client.delete_collection(name="persona_library")
        except:
            pass
            
        persona_collection = chroma_client.create_collection(
            name="persona_library",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 64
            }
        )
        store_personas_in_chroma(PERSONA_LIBRARY)
        print("Persona collection initialized!")
    else:
        print("Using existing persona collection...")
        persona_collection = chroma_client.get_collection(name="persona_library")

    # Restore original logging level
    chromadb_logger.setLevel(original_level)

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
    Stores the given message in the (session-specific) Chroma collection, using the persona name in metadata.
    """
    if SESSION_COLLECTION is None:
        raise ValueError("SESSION_COLLECTION is not initialized.")
    
    embedding = get_openai_embedding(message)
    doc_id = str(uuid.uuid4())  # generate a unique ID

    SESSION_COLLECTION.add(
        documents=[message],
        embeddings=[embedding],
        metadatas=[{"persona": persona_name, "session_id": SESSION_ID}],
        ids=[doc_id]
    )

def store_personas_in_chroma(personas):
    """
    For each persona, embed their description and store in the 'persona_library' collection.
    """
    for p in personas:
        persona_name = p["name"]
        persona_desc = p["desc"]
        persona_sb = p["short_bio"]
        persona_de = ", ".join(p["domain_expertise"])  # Convert list to string
        persona_pt = ", ".join(p["personality_traits"])  # Convert list to string
        persona_rf = p["role_function"]
        persona_exp = p["experience_level"]
        persona_kw = ", ".join(p["style_keywords"])  # Convert list to string

        emb = get_openai_embedding(persona_desc) 
        doc_id = f"persona-{persona_name.lower().replace(' ', '-')}" 
        
        metadata = {
            "persona_name": persona_name,
            "short_bio": persona_sb,
            "domain_expertise": persona_de,
            "personality_traits": persona_pt,
            "role_function": persona_rf,
            "experience_level": persona_exp,
            "style_keywords": persona_kw
        }
        
        persona_collection.add(
            documents=[persona_desc],
            embeddings=[emb],
            metadatas=[metadata],
            ids=[doc_id]
        )

def store_archive_message(persona_name, message):
    """
    Stores message in the 'all_session_archives' collection.
    We call this AFTER the session is done or as the session proceeds.
    """
    global SESSION_ID
    if not SESSION_ID:
        return  # or handle error
    
    emb = get_openai_embedding(message)
    doc_id = str(uuid.uuid4())

    metadata = {
        "session_id": SESSION_ID,
        "persona_name": persona_name
    }

    archive_collection.add(
        documents=[message],
        embeddings=[emb],
        metadatas=[metadata],
        ids=[doc_id]
    )

def search_previous_sessions(user_query, k=5):
    """
    Searches the entire archives for relevant conversation snippets.
    """
    emb = get_openai_embedding(user_query)
    results = archive_collection.query(query_embeddings=[emb], n_results=k)

    # results['documents'] is a list of lists (one per query)
    docs = results['documents'][0] if results and results['documents'] else []
    metas = results['metadatas'][0] if results and results['metadatas'] else []

    # Show them to the user
    print("\n--- Relevant Past Ideas/Sessions ---")
    for i, doc in enumerate(docs):
        meta = metas[i]
        sid = meta.get("session_id")
        persona = meta.get("persona_name")
        print(f"\nMatch {i+1}: from session {sid}, persona {persona}")
        print(f"Snippet: {doc[:200]}...")

def select_personas_by_list():
    # Display all available personas with short bios
    print("\nAvailable Personas:")
    for i, persona in enumerate(PERSONA_LIBRARY, start=1):
        print(f"{i}. {persona['name']} – {persona.get('short_bio', 'No bio available')}")

    # Ask user for indices
    selection = input("\nEnter the indices of the personas you want, separated by commas:\n> ")
    if not selection.strip():
        return []

    try:
        indices = [int(x.strip()) for x in selection.split(",")]
        # Filter out of range
        indices = [i for i in indices if 1 <= i <= len(PERSONA_LIBRARY)]
        chosen_names = [PERSONA_LIBRARY[i-1]["name"] for i in indices]
        return chosen_names
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        return []

def select_personas_by_semantic_search():
    """
    Ask user for a short description, then do a similarity search
    on the persona_library to find the top matches.
    """
    query_desc = input("Describe the type of persona(s) you want:\n> ")

    # Embed the query_desc and query the persona_collection
    query_emb = get_openai_embedding(query_desc)
    results = persona_collection.query(
        query_embeddings=[query_emb],
        n_results=5  # fetch top 5 matches
    )

    # results['ids'] might look like [['persona-rebecca', 'persona-leo', ...]]
    if not results or not results["ids"]:
        print("No matching personas found.")
        return []

    # Flatten the IDs from the first query (since we only used one query)
    top_ids = results["ids"][0]
    top_docs = results["documents"][0]
    top_metadata = results["metadatas"][0]

    # Let's display them
    print("\nTop recommended personas based on your description:\n")
    for i, doc_id in enumerate(top_ids):
        # The metadata might store the persona name. If not, parse from doc_id
        meta = top_metadata[i]
        persona_name = meta.get("persona_name", f"Unknown_{doc_id}")
        short_bio = "No short bio found"
        # Optionally: we can retrieve from the local PERSONA_LIBRARY or store short bio in metadata
        # For simplicity, let's just show the doc excerpt
        excerpt = top_docs[i][:150] + "..." if len(top_docs[i]) > 150 else top_docs[i]

        print(f"{i+1}. {persona_name} – Potential match: {excerpt}")

    # Let user pick which ones they actually want to include
    selection = input("\nEnter the indices of the personas you want, separated by commas:\n> ")
    if not selection.strip():
        return []

    try:
        chosen_indices = [int(x.strip()) for x in selection.split(",")]
        chosen_names = []
        for idx in chosen_indices:
            if 1 <= idx <= len(top_ids):
                chosen_names.append(top_metadata[idx-1].get("persona_name"))
        return chosen_names
    except ValueError:
        print("Invalid input. Returning empty selection.")
        return []

def auto_select_personas_based_on_idea(user_idea):
    """
    Automatically picks the top 3 most relevant personas for the user's idea,
    based on semantic similarity to the idea.
    """
    idea_emb = get_openai_embedding(user_idea)
    results = persona_collection.query(
        query_embeddings=[idea_emb],
        n_results=3  # auto-select top 3
    )

    if not results or not results['metadatas']:
        print("No matching personas found. Defaulting to empty list.")
        return []

    top_metadata = results['metadatas'][0]
    selected_personas = [m['persona_name'] for m in top_metadata]
    print(f"Automatically selected personas: {selected_personas}")
    return selected_personas

PERSONA_CACHE = {}  # { persona_name: persona_desc }

def retrieve_persona_by_name(persona_name: str) -> str:
    """
    Fetches the persona description from the persona_library collection.
    Uses a cache to avoid repeated queries.
    """
    
    if persona_name in PERSONA_CACHE:
        # Already fetched in this session
        return PERSONA_CACHE[persona_name]

    # Perform a naive similarity search:
    emb = get_openai_embedding(persona_name)
    results = persona_collection.query(query_embeddings=[emb], n_results=1)
    
    if results and results['documents']:
        persona_desc = results['documents'][0][0]  # first doc of first query
    else:
        persona_desc = ""
    
    # Store in cache for future lookups
    PERSONA_CACHE[persona_name] = persona_desc
    return persona_desc

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
    Retrieves the top k most relevant documents from the session-specific conversation collection.
    """
    if SESSION_COLLECTION is None:
        return ""
    
    query_embedding = get_openai_embedding(query_text)

    results = SESSION_COLLECTION.query(
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
    # STEP 1: Create a new conversation collection
    create_new_conversation_collection()
    
    # STEP 2: Initialize collections (only recreates what's necessary)
    initialize_persona_collection()

    # STEP 3: Ask user if they want to search old sessions first
    do_search = input("Would you like to search past sessions for inspiration? (y/n)\n> ")
    if do_search.lower().startswith('y'):
        query = input("What would you like to search for?\n> ")
        search_previous_sessions(query)
    
    # Step 4: Ask the user for the idea.
    user_idea = input("What's your idea?\n> ")

    # Step 5: Ask how they want to select personas
    print("How would you like to select personas?")
    print("1. List all available personas and pick any number")
    print("2. Describe what you're looking for, and we'll do a semantic search")
    print("3. Let the system read your idea and automatically select relevant personas")
    choice = input("Enter 1, 2, or 3:\n> ").strip()

    if choice == "1":
        # (A) List + short bios
        selected_personas = select_personas_by_list()
    elif choice == "2":
        # (B) Semantic search
        selected_personas = select_personas_by_semantic_search()
    elif choice == "3":
        # (C) Auto-select based on the idea
        selected_personas = auto_select_personas_based_on_idea(user_idea)
    else:
        print("Invalid choice. Defaulting to listing all personas.")
        selected_personas = select_personas_by_list()

    if not selected_personas:
        print("No personas selected. Exiting.")
        return

    # Step 6: Run the brainstorming loop
    conversation_history = run_brainstorming_with_personas(
        persona_names=selected_personas,
        idea=user_idea,
        total_turns_each=10,
        k=3
    )

    # Step 7: Print out the final conversation in round-robin order
    # Get the number of turns from the first persona's history
    total_turns = len(conversation_history[selected_personas[0]])
    num_personas = len(selected_personas)
    
    # Print full conversation history
    print("\n=== FULL CONVERSATION HISTORY ===")

    # For each turn, find which persona was speaking based on the turn number
    for turn_index in range(total_turns * num_personas):
        current_persona_index = turn_index % num_personas
        persona_name = selected_personas[current_persona_index]
        round_number = turn_index // num_personas
        
        # Only print if this persona still has turns left
        if round_number < len(conversation_history[persona_name]):
            print(f"\n{persona_name}, Turn {round_number + 1}:")
            print(f"{conversation_history[persona_name][round_number]}\n")

    # Step 8. Synthesize final output
    final_output = synthesize_final_output(conversation_history, selected_personas, user_idea)
    print("\n=== FINAL OUTPUT ===")
    print(final_output)

if __name__ == "__main__":
    main()