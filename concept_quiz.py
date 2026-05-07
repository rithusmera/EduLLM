import uuid
import json
import re
import llm_client

LAST_ERROR = None
MAX_CONTEXT_CHARS = 4500


MCQ_PROMPT = """
You are a Physics teacher creating a small concept check quiz.

Generate EXACTLY THREE multiple-choice questions.

STRICT RULES:
- Return ONLY valid JSON
- Do NOT write anything before or after JSON
- Output must start with '{' and end with '}'
- Use double quotes for every key and string
- Do not use trailing commas
- Each question must have options A, B, C, and D
- correct_option must be one of "A", "B", "C", or "D"

Format:

{
 "questions":[
  {
   "difficulty":"easy",
   "question":"...",
   "options":{"A":"...","B":"...","C":"...","D":"..."},
   "correct_option":"A",
   "hint":"..."
  },
  {
   "difficulty":"medium",
   "question":"...",
   "options":{"A":"...","B":"...","C":"...","D":"..."},
   "correct_option":"B",
   "hint":"..."
  },
  {
   "difficulty":"hard",
   "question":"...",
   "options":{"A":"...","B":"...","C":"...","D":"..."},
   "correct_option":"C",
   "hint":"..."
  }
 ]
}

Textbook Context:
{context}
"""


def generate_concept_quiz(context, metadata, max_retries=3):
    global LAST_ERROR
    LAST_ERROR = None

    prompt = MCQ_PROMPT.replace("{context}", trim_context(context))

    for attempt in range(max_retries):
        response = llm_client.run_ollama(prompt)

        try:
            try:
                clean_response = clean_json_string(response)
                parsed = json.loads(clean_response)
            except Exception:
                try:
                    parsed = {"questions": parse_jsonish_quiz(response)}
                except Exception:
                    parsed = {"questions": parse_text_quiz(response)}

            questions = normalize_questions(parsed)

            subject = metadata.get('subject', 'Unknown')
            chapter = metadata.get('chapter', 'Unknown')
            sec_id = metadata.get('section_id', 'Unknown')
            topic_id = f"{subject}|{chapter}|{sec_id}"

            for q in questions:
                q['question_id'] = str(uuid.uuid4())
                q['topic_id'] = topic_id

            return {"questions": questions}

        except Exception as e:
            LAST_ERROR = f"Attempt {attempt + 1}: {e}"
            print(f"[Retry {attempt+1}] Quiz generation error: {e}")
            print(f"Raw Response:\n{response[:2000]}\n")
            continue

    return None


def trim_context(context):
    context = (context or "").strip()
    if len(context) <= MAX_CONTEXT_CHARS:
        return context

    head = context[: MAX_CONTEXT_CHARS // 2]
    tail = context[-MAX_CONTEXT_CHARS // 2 :]
    return f"{head}\n\n[...context trimmed...]\n\n{tail}"


def normalize_questions(parsed):
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        questions = parsed.get("questions", [])
    else:
        raise ValueError("JSON root must be an object with a questions list")

    if len(questions) < 3:
        raise ValueError(f"Expected at least 3 questions, got {len(questions)}")

    normalized = []
    for index, question in enumerate(questions[:3], 1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} is not an object")

        options = question.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"Question {index} is missing options")

        normalized_options = {}
        for letter in ("A", "B", "C", "D"):
            value = options.get(letter) or options.get(letter.lower())
            if not value:
                raise ValueError(f"Question {index} is missing option {letter}")
            normalized_options[letter] = str(value).strip()

        correct_option = str(question.get("correct_option", "")).strip().upper()[:1]
        if correct_option not in normalized_options:
            raise ValueError(f"Question {index} has invalid correct_option")

        question_text = clean_text_fragment(str(question.get("question", "")).strip())
        if not question_text:
            raise ValueError(f"Question {index} is missing question text")

        normalized.append({
            "difficulty": clean_text_fragment(str(question.get("difficulty", "medium")).strip()) or "medium",
            "question": question_text,
            "options": {key: clean_text_fragment(value) for key, value in normalized_options.items()},
            "correct_option": correct_option,
            "hint": clean_text_fragment(str(question.get("hint", "Review the relevant concept in the answer.")).strip()),
        })

    return normalized


def parse_text_quiz(text):
    text = strip_terminal_noise(text)
    blocks = re.findall(
        r"(?:^|\n)\s*(?:Question\s*)?(\d+)\s*[:.)-]\s*(.*?)(?=\n\s*(?:Question\s*)?\d+\s*[:.)-]|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    questions = []

    for _, block in blocks:
        question = parse_text_question_block(block)
        if question:
            questions.append(question)

    if len(questions) < 3:
        raise ValueError(f"No valid JSON and only found {len(questions)} text questions")

    return questions[:3]


def parse_jsonish_quiz(text):
    text = strip_terminal_noise(text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")

    chunks = re.split(r'(?="\s*difficulty"\s*:)', text)
    questions = []

    for chunk in chunks:
        if '"question"' not in chunk or '"options"' not in chunk:
            continue

        question_text = extract_jsonish_string(chunk, "question")
        correct_option = extract_jsonish_string(chunk, "correct_option").upper()[:1]
        hint = extract_jsonish_string(chunk, "hint")
        difficulty = extract_jsonish_string(chunk, "difficulty") or "medium"
        options_match = re.search(r'"options"\s*:\s*\{(.*?)\}', chunk, flags=re.DOTALL)
        if not options_match:
            continue

        options = {}
        for letter, value in re.findall(r'"([A-D])"\s*:\s*"([^"]*)"', options_match.group(1), flags=re.DOTALL):
            options[letter.upper()] = clean_text_fragment(value)

        if question_text and len(options) >= 4 and correct_option in options:
            questions.append({
                "difficulty": difficulty,
                "question": question_text,
                "options": options,
                "correct_option": correct_option,
                "hint": hint or "Review the relevant concept from the answer.",
            })

    if len(questions) < 3:
        raise ValueError(f"Only found {len(questions)} JSON-like questions")

    return questions[:3]


def extract_jsonish_string(text, key):
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    if not match:
        return ""

    return clean_text_fragment(match.group(1))


def clean_text_fragment(text):
    words = " ".join((text or "").split()).split()
    cleaned = []
    index = 0

    while index < len(words):
        current = words[index]
        next_word = words[index + 1] if index + 1 < len(words) else ""
        current_key = re.sub(r"\W+", "", current).lower()
        next_key = re.sub(r"\W+", "", next_word).lower()

        if next_key and (current_key == next_key or (len(current_key) >= 3 and next_key.startswith(current_key))):
            index += 1
            continue

        cleaned.append(current)
        index += 1

    return " ".join(cleaned)


def parse_text_question_block(block):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    question_lines = []
    options = {}
    correct_option = ""
    hint = ""
    current_option = None

    for line in lines:
        option_match = re.match(r"^([A-D])\s*[\).:-]\s*(.+)$", line, flags=re.IGNORECASE)
        correct_match = re.match(r"^Correct\s*(?:Option|Answer)?\s*[:\-]\s*([A-D])", line, flags=re.IGNORECASE)
        hint_match = re.match(r"^Hint\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)

        if correct_match:
            correct_option = correct_match.group(1).upper()
            current_option = None
        elif hint_match:
            hint = hint_match.group(1).strip()
            current_option = None
        elif option_match:
            current_option = option_match.group(1).upper()
            options[current_option] = option_match.group(2).strip()
        elif current_option:
            options[current_option] = f"{options[current_option]} {line}".strip()
        else:
            question_lines.append(line)

    question_text = " ".join(question_lines).strip()
    if not question_text or len(options) < 4 or correct_option not in options:
        return None

    return {
        "difficulty": "medium",
        "question": question_text,
        "options": options,
        "correct_option": correct_option,
        "hint": hint or "Review the explanation and compare each option with the concept.",
    }


def strip_terminal_noise(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text or "")
    text = re.sub(r"\\\s*(?=\r?\n)", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return text

def clean_json_string(text):
    """
    Robust JSON cleaner for LLM output
    """

    text = strip_terminal_noise(text).strip()
    if not text:
        raise ValueError("Model returned an empty response")

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    text = fix_multiline_json_strings(text)

    repaired = repair_jsonish_text(text)
    if repaired:
        return repaired

    decoder = json.JSONDecoder()
    starts = [i for i, char in enumerate(text) if char in "[{"]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, list) or (isinstance(obj, dict) and "questions" in obj):
                return json.dumps(obj)
        except json.JSONDecodeError:
            continue

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    json_str = match.group(0)

    json_str = fix_multiline_json_strings(json_str)
    json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    json_str = re.sub(r'\s+', ' ', json_str)

    return json_str.strip()


def fix_multiline_json_strings(text):
    def fix_multiline_strings(string_match):
        return string_match.group(0).replace("\n", " ").replace("\r", " ")

    return re.sub(r'"(?:\\.|[^"\\])*"', fix_multiline_strings, text, flags=re.DOTALL)


def repair_jsonish_text(text):
    if '"questions"' not in text:
        return None

    start = text.find("{")
    array_end = text.rfind("]")
    if start == -1 or array_end == -1 or array_end <= start:
        return None

    candidate = text[start:array_end + 1]
    candidate = f"{candidate}}}"
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, dict) and "questions" in obj:
        return json.dumps(obj)

    return None

def grade_mcq(question_obj, selected_option):
    return 1 if selected_option == question_obj['correct_option'] else 0
