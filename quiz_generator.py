import uuid
import json
import llm_client

MCQ_PROMPT = """
You are a Physics teacher generating a multiple-choice question based on the textbook context provided.

Output ONLY a valid JSON object. Do not include any introductory text, markdown formatting (like ```json), or explanations
IMPORTANT: If using LaTeX or backslashes, escape them properly for JSON (use \\\\).

The JSON object must have these keys:
- "question": The question text.
- "options": A dictionary with keys "A", "B", "C", "D".
- "correct_option": The letter "A", "B", "C", or "D".
- "hint": A small hint based on the context.

Textbook Context:
------------------
{context}
"""

def generate_mcq(context, metadata, max_retries = 3):
    prompt = MCQ_PROMPT.format(context = context)

    for attempt in range(max_retries): #If first attempt to get a response in proper json format fails, try again
        response = llm_client.run_ollama(prompt) #LLM return response in json format
        clean_response = clean_json_string(response)
        
        try:
            parsed = json.loads(clean_response)

            subject = metadata.get('subject', 'Unknown')
            chapter = metadata.get('chapter', 'Unknown')
            sec_id = metadata.get('section_id', 'Unknown')

            topic_id = f"{subject}|{chapter}|{sec_id}" #Each section is one topic

            parsed['question_id'] = str(uuid.uuid4())
            parsed['topic_id'] = topic_id
            
            return parsed
        
        except Exception as e:
            print(f"Parsing Error: {e} | Raw Response: {response}")
            continue

def clean_json_string(text):
    '''Remove any extra text before and after the json string'''

    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            json_str = json_str.replace('\\\n', '\n').replace('\\', '')
            json_str = json_str.replace('```json', '').replace('```', '').strip()

            return json_str
        
    except Exception as e:
        print(f"Cleaning error: {e}")
    
    return text.strip()

def grade_mcq(question_obj, selected_option):
    return 1 if selected_option == question_obj['correct_option'] else 0