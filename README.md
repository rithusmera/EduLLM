# EduLLM

EduLLM is an offline AI tutor for Class 11–12 science students, built with Streamlit and a local RAG (Retrieval-Augmented Generation) pipeline over textbook content. It runs entirely on-device using a local Ollama model, so no internet connection or API key is required to ask questions or generate quizzes.

## Features

- **Ask a question** — Type any question about the syllabus and get an answer grounded in the actual textbook, using semantic search (FAISS + Sentence Transformers) to pull the most relevant chunks. Direct references like "Figure 3.2", "Table 1.1", or "Example 4" are detected and retrieved exactly, along with their parent section for context.
- **Concept quiz** — After getting an answer, generate a 3-question multiple-choice quiz (easy/medium/hard) based on that specific answer's context, with hints and instant grading.
- **Chapter quiz** — Pick a subject and chapter to take a pre-authored quiz sampled from a local question bank (`quiz_bank/`), with a mix of easy, medium, and hard questions.
- **Dynamic study plan** — Every quiz attempt is logged to a local database and used to compute a per-topic mastery score (via exponential smoothing). The sidebar surfaces the topics you're weakest on, with a priority level and a recommendation.
- **User accounts** — Simple username/password registration and login (SHA-256 hashed, stored locally in SQLite).
- **Theming** — Pick an accent color from presets or a custom color picker; the whole UI re-themes live.
- **Math rendering** — LaTeX-style math delimiters (`\(...\)`, `\[...\]`, `equation`/`align` environments) are normalized and rendered with KaTeX via Streamlit.

## How it works

1. **Chunking** (`Chunking.py`) — Splits a plain-text chapter file into sections/sub-items (text, examples, tables, figures, exercises) with overlap-based sliding-window chunking for long passages.
2. **Embedding** (`Embedding.py`) — Encodes each chunk with the `all-MiniLM-L6-v2` sentence-transformer model and writes the vectors out to JSON.
3. **Vector DB build** (`CreateVectorDB.py`) — Loads the chunks and embeddings, stores chunk metadata/content in a SQLite database (`edu_chunks.db`), and adds the normalized vectors to a FAISS index (`edu_index.faiss`).
4. **Retrieval** (`RAGPipeline.py`) — At query time, either matches a direct reference (figure/table/example/exercise number) or does a FAISS similarity search, then pulls the matching chunks (plus parent sections) from SQLite to build the context for the LLM.
5. **Generation** (`llm_client.py`) — Sends the assembled prompt to a locally running `ollama` process (default model: `mistral`) and cleans up terminal control characters from the streamed output.
6. **Quizzing** (`concept_quiz.py`, `chapter_quiz.py`) — The concept quiz asks the LLM to generate strict JSON MCQs from the last answer's context (with several fallback parsers for malformed model output); the chapter quiz samples from static JSON question banks under `quiz_bank/<subject>/`.
7. **Mastery tracking** (`student_state.py`) — Every graded attempt updates a per-student, per-topic mastery score and feeds the study-plan recommendations shown in the sidebar.

## Project structure

```
EduLLM/
├── app.py                        # Main Streamlit page: ask a question, concept quiz, study plan
├── auth.py                       # Login-gate helper for protected pages
├── login.py                      # User registration/login (SQLite + SHA-256)
├── theme.py                      # Accent color theming + global CSS
├── ui_format.py                  # Math delimiter normalization + markdown/LaTeX rendering
├── RAGPipeline.py                # FAISS search + chunk/parent-section retrieval (+ CLI test harness)
├── llm_client.py                 # Local Ollama subprocess wrapper
├── concept_quiz.py               # On-the-fly concept quiz generation from LLM output
├── chapter_quiz.py               # Static chapter quiz loading, filtering, sampling
├── student_state.py              # SQLite-backed attempt logging + mastery scoring
├── Chunking.py                   # Textbook .txt -> structured chunk JSON
├── Embedding.py                  # Chunk JSON -> embedding JSON
├── CreateVectorDB.py             # Embedding JSON -> FAISS index + SQLite chunk store
├── EmbeddingBasedRetrieval_test.py  # Standalone cosine-similarity retrieval test script
├── edu_chunks.db                 # Pre-built chunk store (SQLite)
├── edu_index.faiss               # Pre-built FAISS index
├── pages/
│   ├── login_page.py              # Sign in / register UI
│   └── chapter_quiz_page.py       # Chapter quiz UI
└── quiz_bank/
    └── physics/
        ├── chapter1.json
        └── chapter2.json
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with a model pulled (default is `mistral`):
  ```bash
  ollama pull mistral
  ```

## Installation

```bash
git clone https://github.com/rithusmera/EduLLM.git
cd EduLLM
pip install streamlit faiss-cpu sentence-transformers numpy scikit-learn
```

## Running the app

```bash
streamlit run app.py
```

Register an account on first launch, sign in, and start asking questions on the "Ask a question" page. Use the sidebar to switch to the Chapter Quiz page or change the accent theme.

## Building your own textbook index

The included `edu_chunks.db` and `edu_index.faiss` are pre-built from a Physics Class 11 chapter. To index your own content:

1. Update the `CLASS`, `SUBJECT`, `CHAPTER`, and `file_path` constants at the top of `Chunking.py`, then run it to produce a chunked JSON file.
2. Update the input/output paths in `Embedding.py` and run it to generate embeddings for those chunks.
3. Update the paths in `CreateVectorDB.py` and run it to append the new chunks and vectors into `edu_chunks.db` / `edu_index.faiss`.

To add chapter-quiz questions for a new subject, add a folder under `quiz_bank/<subject>/` containing JSON files shaped like `{"questions": [...]}`, where each question has `subject`, `chapter`, `difficulty` (1/2/3), `question`, `options`, `correct_option`, and `hint` fields.

## Notes

- All inference runs locally through Ollama — no external API calls are made for answering questions or generating quizzes.
- Student progress (`student_state.db`) and user accounts (`users.db`) are stored in local SQLite files created on first run.
