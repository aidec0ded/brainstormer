# Brainstorming App

A Python-based application that facilitates **multi-persona** brainstorming sessions using **OpenAI** and **ChromaDB**. This app allows you to:

- Dynamically select or auto-select personas for brainstorming.
- Store and retrieve conversation history in **ChromaDB**.
- Generate a final **consulting-style** proposal based on the conversation.

## Features

1. **Persona Library**  
   - Includes 25 unique personas (in `personas.py`) spanning various roles: AI ethicists, hardware engineers, legal experts, marketing strategists, etc.

2. **Multiple Persona Selection Methods**  
   - **Manual List Selection**: Pick from a displayed list of personas.  
   - **Semantic Search**: Type a short description of the persona type you need, and the system suggests best matches.  
   - **Auto Selection**: The application reads your idea and automatically picks the top personas based on similarity.

3. **Conversation Storage**  
   - All generated responses are embedded and stored in ChromaDB for context retrieval.  
   - Each new session starts fresh by clearing or resetting the conversation collection, ensuring no carryover from previous runs.

4. **Final Output**  
   - After the conversation, a final proposal is generated summarizing the idea, featuring an executive summary, recommended roadmap, and more.

## Quick Start

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aidec0ded/brainstormer.git
   cd brainstormer
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the Application**
   ```bash
   python app.py
You will be prompted for your idea, how to select personas, etc.
4. **(Optional) Set OpenAI API Key**
   * You may need to provide your OpenAI API key as an environmental variable
     ```bash
     export OPENAI_API_KEY="sk-..."
  * Or inject it into your code where openai.api_key is set.

**Project Structure**
* app.py
Main application logic (contains `main()` function, conversation flow).
* personas.py
Contains the PERSONA_LIBRARY with extended metadata.
* requirements.txt
Python dependencies.
* .gitignore
Rules to exclude environment files, Chroma DB data, etc.
* README.md
You’re reading it!

**Customization**
* **Adding Personas**: Update `personas.py` with your new entries (name, desc, domain expertise, etc.).
* **Changing LLM Model**: Adjust the model parameter in app.py for your `ChatCompletion` calls.
* **Modifying Summaries**: If you want shorter or more detailed final output, tweak your prompt in `synthesize_final_output`.