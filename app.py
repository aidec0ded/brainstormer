from openai import OpenAI
import chromadb
import uuid 
import datetime
import logging
import re
import json
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
                "hnsw:construction_ef": 250,
                "hnsw:search_ef": 150,
                "hnsw:M": 32
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
        # Convert lists to comma-separated strings for metadata
        metadata = {
            "persona_name": persona_name,
            "short_bio": p["short_bio"],
            "domain_expertise": ", ".join(p["domain_expertise"]),      # Convert list to string
            "personality_traits": ", ".join(p["personality_traits"]),  # Convert list to string
            "role_function": p["role_function"],
            "experience_level": p["experience_level"],
            "style_keywords": ", ".join(p["style_keywords"])          # Convert list to string
        }

        emb = get_openai_embedding(p["desc"]) 
        doc_id = f"persona-{persona_name.lower().replace(' ', '-')}" 
        
        persona_collection.add(
            documents=[p["desc"]],
            embeddings=[emb],
            metadatas=[metadata],
            ids=[doc_id]
        )

def store_persona_fields_in_chroma(personas):
    """
    For each persona, embed relevant fields separately and store them in 'persona_library'.
    Each field is a separate record, letting us do field-specific searches.
    """
    for p in personas:
        persona_name = p["name"]
        
        # We'll store these fields separately
        fields_to_embed = {
            "desc": p["desc"],
            "short_bio": p["short_bio"],
            "domain_expertise": ", ".join(p["domain_expertise"]),
            "personality_traits": ", ".join(p["personality_traits"]),
            "role_function": p["role_function"],
            "experience_level": p["experience_level"],
            "style_keywords": ", ".join(p["style_keywords"])
        }
        
        for field_name, field_text in fields_to_embed.items():
            # Skip if empty
            if not field_text:
                continue
            
            emb = get_openai_embedding(field_text)
            doc_id = f"persona-{persona_name.lower().replace(' ', '-')}-{field_name}"
            
            metadata = {
                "persona_name": persona_name,
                "field_name": field_name,
            }
            # We might also carry the field_text in 'documents' to reconstruct or debug
            persona_collection.add(
                documents=[field_text],
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

def store_persona_learned_embedding(persona_name, conversation_history):
    """
    Summarizes how a persona performed or evolved in this session, 
    then stores a new 'learned embedding' for them in 'persona_library'.
    """
    # 1) Generate a summary or reflection
    # We can pass the conversation_history specifically for this persona
    persona_dialogue = conversation_history.get(persona_name, [])
    dialogue_text = "\n\n".join(persona_dialogue)

    system_prompt = (
        "You are an analyzer for persona evolution. The user has a conversation where this persona participated. "
        "Summarize how the persona expressed themselves, new insights, or unique traits that emerged. "
        "Output a refined persona summary, focusing on new knowledge or style changes observed in the conversation."
    )

    prompt_messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": f"Persona Name: {persona_name}\n\nConversation:\n{dialogue_text}"}
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt_messages,
        max_tokens=500,
        temperature=0.7
    )

    learned_summary = completion.choices[0].message.content.strip()

    # 2) Embed and store in persona_library with a special doc_id
    learned_emb = get_openai_embedding(learned_summary)
    doc_id = f"persona-{persona_name.lower().replace(' ', '-')}-learned-{SESSION_ID}"

    metadata = {
        "persona_name": persona_name,
        "learned_from_session": SESSION_ID,
        "field_name": "learned_summary"
    }

    persona_collection.add(
        documents=[learned_summary],
        embeddings=[learned_emb],
        metadatas=[metadata],
        ids=[doc_id]
    )

    print(f"Stored learned embedding for {persona_name} from session {SESSION_ID}.\nSummary:\n{learned_summary}\n")

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

def parse_domains_from_manager_output(manager_text: str) -> list:
    """
    Attempts to parse the manager agent's textual output for domain expertise or roles.
    
    The manager agent might respond in multiple formats:
    - JSON list, e.g.: ["AI Ethics", "Hardware Engineering"]
    - Simple text listing domain(s)
    - A bullet-point list
    This function tries to handle these common formats and return a Python list of strings.
    
    If no recognizable domains are found, returns an empty list.
    """
    manager_text = manager_text.strip()
    
    # 1) Try to interpret the entire text as JSON
    #    e.g. the manager might have responded with exactly ["AI Ethics","Hardware Engineering"]
    try:
        possible_list = json.loads(manager_text)
        if isinstance(possible_list, list):
            # Make sure each item is a string
            domain_list = [str(item).strip() for item in possible_list]
            # Filter out empties
            return [d for d in domain_list if d]
    except:
        pass  # Not valid JSON or the manager gave a more free-form text
    
    # 2) If not valid JSON, search for bullet points or lines
    #    e.g. "1) AI Ethics\n2) Hardware Engineering"
    lines = manager_text.splitlines()
    possible_domains = []
    for line in lines:
        # Remove numbering or bullet points
        line = line.strip().lstrip("0123456789) .-").strip()
        # If line is short or doesn't contain typical domain words, skip it
        # This is optional – up to you how lenient you want to be
        if len(line) > 2:
            possible_domains.append(line)
    
    if possible_domains:
        return possible_domains
    
    # 3) As a fallback, do a simple regex search for bracketed or quoted items
    #    e.g. "Something like [AI Ethics, Hardware Engineering]"
    bracket_pattern = r"\[([^\]]+)\]"
    match = re.search(bracket_pattern, manager_text)
    if match:
        inside_brackets = match.group(1).split(",")
        domain_list = [d.strip() for d in inside_brackets]
        return [d for d in domain_list if d]

    # 4) If no patterns matched, return empty
    return []

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

def manager_agent_select_personas(user_idea: str, all_personas: list, top_k=5):
    """
    Asks GPT-4 to figure out which domains/roles are needed for the user's idea.
    Then queries 'persona_library' for best matching personas by domain_expertise or role_function.
    Returns a list of persona names.
    """
    # 1) Summarize or label the user_idea
    manager_prompt = [
        {
            "role": "developer",
            "content": (
                "You are a 'manager agent' that analyzes a new idea and decides which roles "
                "or expertise are crucial to evaluate and develop it. "
                "You'll output a JSON list of relevant domain_expertise or role_functions."
            )
        },
        {
            "role": "user",
            "content": (
                f"User Idea:\n{user_idea}\n\n"
                "Identify which 3-5 domain_expertise or role_functions are most relevant for exploring or executing this idea. "
                "Return them as JSON, e.g.:\n"
                '[\n  "AI Ethics",\n  "Hardware Engineering"\n]\n'
            )
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=manager_prompt,
        max_tokens=300,
        temperature=0.4
    )

    # 2) Parse the manager agent's output
    manager_response = completion.choices[0].message.content.strip()
    print(f"Manager Agent's Raw Output:\n{manager_response}\n")

    import json
    try:
        needed_domains = json.loads(manager_response)
        if not isinstance(needed_domains, list):
            needed_domains = []
    except:
        needed_domains = []
    
    if not needed_domains:
        # Fallback: if manager agent doesn't parse well, just return empty or let user pick
        print("Manager agent did not return a valid list. No domain_expertise found.")
        return []
    
    # 3) We search the 'persona_library' for these domains
    results = persona_collection.query(
        query_texts=["Selecting persona for the user's idea"],
        n_results=10,
        where={
            "domain_expertise": {"$in": needed_domains}
        }
    )

    # Extract persona names from the results
    matching_personas = []
    if results and results['metadatas']:
        matching_personas = [
            meta['persona_name'] 
            for meta in results['metadatas'][0]  # First query's metadata
        ]

    # If we found too many, limit to top_k by also doing a similarity search
    if len(matching_personas) > top_k:
        matching_personas = matching_personas[:top_k]

    print(f"Manager Agent suggested: {matching_personas}")
    return matching_personas


def manager_agent_create_persona_if_needed(user_idea: str, required_domains: list) -> list:
    """
    1) Attempt to find personas with these required_domains in metadata.
    2) If none found, create new personas to fill the gaps.
    3) Return list of persona names to use (both existing and new).
    """
    # First try to find some existing relevant personas via semantic search
    query_text = f"Expert in: {', '.join(required_domains)}"
    results = persona_collection.query(
        query_embeddings=[get_openai_embedding(query_text)],
        n_results=3
    )
    
    existing_personas = []
    if results and results['metadatas']:
        existing_personas = [meta['persona_name'] for meta in results['metadatas'][0]]
    
    if existing_personas:
        print(f"Found existing relevant personas: {existing_personas}")
    
    # If we don't have enough personas, create new ones
    if len(existing_personas) < 2:  # Ensure we have at least 2 personas
        print("Creating new personas to complement the conversation...")
        
        # Determine how many new personas to create (2-3 total)
        num_new_needed = min(3 - len(existing_personas), len(required_domains))
        
        creation_prompt = [
            {
                "role": "developer",
                "content": (
                    "You are a persona creator for a collaborative brainstorming system. "
                    "Create unique personas with different expertise and perspectives that would be valuable "
                    f"for discussing this idea: {user_idea}\n\n"
                    f"Create {num_new_needed} different personas, each specializing in different aspects "
                    f"of these domains: {', '.join(required_domains)}\n\n"
                    "Return a JSON array of personas, each with these fields:\n"
                    "- name: A memorable, realistic name\n"
                    "- short_bio: A one-line bio\n"
                    "- desc: A detailed description of their expertise and perspective (2-3 paragraphs)\n"
                    "- domain_expertise: List of their specific expertise areas\n"
                    "- personality_traits: List of 3-5 defining traits\n"
                    "- role_function: Their primary professional role\n"
                    "- experience_level: Senior, Mid-level, or Expert\n"
                    "- style_keywords: List of words that characterize their communication style\n\n"
                    "Make each persona distinct and specialized, with clear areas of expertise."
                )
            }
        ]
        
        completion = client.chat.completions.create(
            model="gpt-4",  # Using most capable model for persona creation
            messages=creation_prompt,
            max_tokens=2000,
            temperature=0.7
        )

        try:
            new_personas = json.loads(completion.choices[0].message.content)
            for persona in new_personas:
                store_new_persona_in_chroma(persona)
                existing_personas.append(persona["name"])
        except json.JSONDecodeError as e:
            print(f"Error parsing persona JSON: {e}")
            print("Raw response:", completion.choices[0].message.content)
            # Instead of using fallback persona, we could retry or raise an error
            raise ValueError("Failed to create valid personas")

    return existing_personas

def create_gap_filling_persona(user_idea: str, required_domains: list) -> list:
    """
    Similar to manager_agent_create_persona_if_needed but specifically for filling gaps
    during conversation. Only creates one persona at a time if needed.
    """
    # First try to find an existing relevant persona via semantic search
    query_text = f"Expert in: {', '.join(required_domains)}"
    results = persona_collection.query(
        query_embeddings=[get_openai_embedding(query_text)],
        n_results=1
    )
    
    if results and results['metadatas']:
        persona_name = results['metadatas'][0][0]['persona_name']
        print(f"Found existing relevant persona for gap: {persona_name}")
        return [persona_name]
    
    # If no existing persona found, create a new one
    print("Creating new persona to fill expertise gap...")
    
    creation_prompt = [
        {
            "role": "developer",
            "content": (
                "You are a persona creator for a collaborative brainstorming system. "
                "Create a single unique persona with expertise to fill this gap in the conversation. "
                f"The original idea being discussed is: {user_idea}\n\n"
                f"The persona should specialize in these domains: {', '.join(required_domains)}\n\n"
                "Return a JSON object with these fields:\n"
                "- name: A memorable, realistic name\n"
                "- short_bio: A one-line bio\n"
                "- desc: A detailed description of their expertise and perspective (2-3 paragraphs)\n"
                "- domain_expertise: List of their specific expertise areas\n"
                "- personality_traits: List of 3-5 defining traits\n"
                "- role_function: Their primary professional role\n"
                "- experience_level: Senior, Mid-level, or Expert\n"
                "- style_keywords: List of words that characterize their communication style\n\n"
                "Make the persona specialized and focused on filling the identified gap."
            )
        }
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=creation_prompt,
        max_tokens=2000,
        temperature=0.7
    )

    try:
        new_persona = json.loads(completion.choices[0].message.content)
        store_new_persona_in_chroma(new_persona)
        return [new_persona["name"]]
    except json.JSONDecodeError as e:
        print(f"Error parsing persona JSON: {e}")
        print("Raw response:", completion.choices[0].message.content)
        raise ValueError("Failed to create valid persona")

def find_personas_by_domains(domains: list, top_k=5) -> list:
    """
    Returns a list of persona names that match any of the domains in 'domains'.
    Using Chroma's metadata filter with '$contains'.
    """
    if not domains:
        return []
    
    # Query using the OpenAI embedding and a simpler where clause
    results = persona_collection.query(
        query_embeddings=[get_openai_embedding("Retrieving persona by domain expertise")],
        where={
            "$or": [
                {
                    "domain_expertise": {
                        "$in": [domain.lower()]  # Case-insensitive matching
                    }
                } 
                for domain in domains
            ]
        },
        n_results=top_k
    )

    matching_personas = []
    if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
        for meta in results['metadatas'][0]:
            matching_personas.append(meta['persona_name'])
    return list(set(matching_personas))  # unique


def store_new_persona_in_chroma(persona_dict):
    """
    Takes a newly minted persona dict and stores it in both ChromaDB and the PERSONA_LIBRARY file.
    """
    # ... [previous ChromaDB storage code remains the same] ...

    # Append to PERSONA_LIBRARY file
    personas_file_path = "personas.py"
    with open(personas_file_path, 'r') as file:
        content = file.read()
    
    # Find the last entry and ensure it ends with a comma
    if '}]' in content:
        content = content.replace('}]', '},')
    elif '}\n]' in content:
        content = content.replace('}\n]', '},')

    # Format the new persona as a dictionary string
    new_persona_str = f"""    {{
        'name': {repr(persona_dict["name"])},
        'short_bio': {repr(persona_dict["short_bio"])},
        'desc': {repr(persona_dict["desc"])},
        'domain_expertise': {repr(persona_dict["domain_expertise"])},
        'personality_traits': {repr(persona_dict["personality_traits"])},
        'role_function': {repr(persona_dict["role_function"])},
        'experience_level': {repr(persona_dict["experience_level"])},
        'style_keywords': {repr(persona_dict["style_keywords"])}
    }}\n]"""

    # Write the updated content
    with open(personas_file_path, 'w') as file:
        file.write(content)
        file.write(new_persona_str)

    print(f"New Persona '{persona_dict['name']}' created and stored in both ChromaDB and PERSONA_LIBRARY.\n")

def manager_agent_decide_personas(user_idea):
    """
    Manager agent logic that decides which domain_expertise are needed,
    then calls manager_agent_create_persona_if_needed.
    Returns a list of persona names to use.
    """
    # Basic manager agent approach:
    manager_prompt = [
        {"role": "system", "content": "You are a manager agent deciding domain expertise needed."},
        {"role": "user", "content": f"User idea:\n{user_idea}\n\nWhich 2-3 domains are needed?"}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=manager_prompt,
        max_tokens=200,
        temperature=0.6
    )
    # parse manager response as a list of domains
    domain_list = parse_domains_from_manager_output(completion.choices[0].message.content)

    # Now create or fetch personas
    persona_names = manager_agent_create_persona_if_needed(user_idea, domain_list)
    return persona_names

def manager_agent_monitor_conversation(conversation_history, persona_names, user_idea):
    """
    Looks at the last round of conversation, checks if there's a domain gap.
    If there's a gap, create/inject a new persona.
    Returns possibly updated persona_names if we add a new one.
    """
    # First ensure all personas have conversation history entries
    for persona in persona_names:
        if persona not in conversation_history:
            conversation_history[persona] = []
    
    # Get last responses, but only for personas who have spoken
    last_responses = [conversation_history[p][-1] for p in persona_names if conversation_history[p]]
    
    # If no responses yet, return without changes
    if not last_responses:
        return persona_names
    
    # feed that into an LLM prompt
    monitor_prompt = [
        {"role": "system", "content": "You are a gap-detecting manager agent."},
        {"role": "user", "content": (
            f"User idea: {user_idea}\n\n"
            f"Recent persona responses:\n{last_responses}\n\n"
            "Do we have any domain gaps? If so, name them or say 'No Gap' if everything is covered."
        )}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=monitor_prompt,
        max_tokens=300,
        temperature=0.6
    )

    gap_report = completion.choices[0].message.content.strip()
    if "No Gap" in gap_report:
        return persona_names  # no change

    # Otherwise parse gap_report into domain list
    new_domains = parse_domains_from_manager_output(gap_report)
    if not new_domains:
        return persona_names

    # create new persona if none exist
    new_personas = create_gap_filling_persona(user_idea, new_domains)
    # add them to persona_names and initialize their conversation history
    for new_persona in new_personas:
        if new_persona not in persona_names:
            persona_names.append(new_persona)
            conversation_history[new_persona] = []
            
    return persona_names

def reasoning_agent_review(conversation_history, persona_names):
    """
    The reasoning agent reads the entire conversation so far,
    highlights contradictions or suggestions to refine.
    Returns a short string summarizing them.
    """
    # Convert conversation_history into a text block
    transcript = ""
    for pn in persona_names:
        # Just the last response or the entire conversation, up to you
        for i, resp in enumerate(conversation_history[pn], start=1):
            transcript += f"{pn} (turn {i}): {resp}\n"

    # Now call GPT-4 to find contradictions, improvements
    agent_prompt = [
        {
            "role": "system",
            "content": (
                "You are a reasoning agent. Your job is to spot inconsistencies, "
                "contradictions, or areas needing further exploration. Provide concise bullet points."
            )
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript}\n\n"
                "List any contradictions or improvements that should be addressed next."
            )
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=agent_prompt,
        max_tokens=400,
        temperature=0.5
    )

    critique = completion.choices[0].message.content.strip()
    return critique

PERSONA_CACHE = {}  # { persona_name: persona_desc }

def retrieve_persona_by_name(persona_name: str) -> str:
    """
    Fetches the persona description from the persona_library collection.
    Uses a cache to avoid repeated queries.
    """
    
    if persona_name in PERSONA_CACHE:
        # Already fetched in this session
        return PERSONA_CACHE[persona_name]

    # 1) Find the doc with field_name=desc
    # 2) Also find docs with field_name=learned_summary
    # 3) Combine them
    # We'll do a naive approach here for brevity:
    query_text = f"{persona_name} learned_summary desc"
    emb = get_openai_embedding(query_text)
    results = persona_collection.query(
        query_embeddings=[emb],
        n_results=5
    )

    # parse out the best match for "desc" and also any "learned_summary"
    # for simplicity, just combine them
    combined_desc = ""
    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            if meta.get("persona_name") == persona_name:
                # e.g. if "field_name" in ["desc","learned_summary"]
                if meta.get("field_name") in ("desc", "learned_summary"):
                    combined_desc += doc + "\n---\n"
    
    # Store in cache for future lookups
    PERSONA_CACHE[persona_name] = combined_desc
    return combined_desc

def run_brainstorming_with_reasoning(persona_names, idea, total_turns_each=10, k=3):
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

        # Reasoning agent critique so far
        critique = reasoning_agent_review(conversation_history, persona_names)

        # Combine everything in the 'context'
        combined_context = (
            f"{relevant_context}\n\n"
            f"Reasoning Agent Critique:\n{critique}"
        )
        
        # Generate persona’s response with their “essence”
        next_response = generate_response_for_persona(persona_name, idea, relevant_context)
        
        # Store in local history + vector DB
        conversation_history[persona_name].append(next_response)
        store_message_in_chroma(persona_name, next_response)

        # After each complete round (when all personas have spoken), check for gaps
        if (turn_index + 1) % num_personas == 0:
            updated_persona_names = manager_agent_monitor_conversation(conversation_history, persona_names, idea)
            if len(updated_persona_names) > len(persona_names):
                # New persona(s) were added
                persona_names = updated_persona_names
                num_personas = len(persona_names)
                total_turns = num_personas * total_turns_each
                # Initialize conversation history for new personas
                for name in persona_names:
                    if name not in conversation_history:
                        conversation_history[name] = []

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
            # Get persona description from ChromaDB instead of PERSONA_LIBRARY
            persona_desc = retrieve_persona_by_name(persona_name)
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
                "Use the conversation transcript and the user's original idea as context."
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
    print("3. Manager Agent picks for me.")
    choice = input("Enter 1, 2, or 3:\n> ").strip()

    if choice == "1":
        # (A) List + short bios
        selected_personas = select_personas_by_list()
    elif choice == "2":
        # (B) Semantic search
        selected_personas = select_personas_by_semantic_search()
    elif choice == "3":
        # (C) Auto-select based on the idea
        selected_personas = manager_agent_decide_personas(user_idea)
    else:
        print("Invalid choice. Defaulting to listing all personas.")
        selected_personas = select_personas_by_list()

    if not selected_personas:
        print("No personas selected. Exiting.")
        return

    # Step 6: Run the brainstorming loop
    conversation_history = run_brainstorming_with_reasoning(
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

    # Step 9: For each persona in the session, store a learned embedding
    for persona_name in selected_personas:
        store_persona_learned_embedding(persona_name, conversation_history)
    
    print("\n=== FINAL OUTPUT ===")
    print(final_output)

if __name__ == "__main__":
    main()