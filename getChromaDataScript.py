import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

conversation = client.get_collection(name="conversation_history")
personas = client.get_collection(name="persona_library")

start_convo = conversation.peek()
top_personas = personas.peek()

print(start_convo)
print("-----")
print(top_personas)