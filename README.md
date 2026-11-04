# MatrixMind — Linear Algebra AI Workspace

**MatrixMind** is a browser-based linear algebra learning environment combining a numerical matrix workbench, Retrieval-Augmented Generation (RAG), and a subject-specific Knowledge Graph.

## Main features

### 1. Linear Algebra Matrix Lab

The Matrix Lab provides validated numerical operations through NumPy:

- Matrix addition, subtraction, and multiplication
- Transpose, determinant, inverse, rank, and trace
- Eigenvalues
- Reduced row echelon form (RREF)
- Solve a square system `Ax = b`
- Frobenius norm

The API checks matrix dimensions and reports clear errors for invalid or undefined operations. Each calculation returns a short explanation of the method.

### 2. RAG-based AI Tutor

Ask questions about linear algebra in natural language. MatrixMind:

1. Embeds the student's question using a lightweight, local feature-hashing vectorizer.
2. Retrieves relevant text chunks from indexed notes using vector similarity and lexical overlap.
3. Expands the retrieved context through connected graph concepts when graph-enhanced retrieval is enabled.
4. Sends the retrieved context to Groq for a generated, source-grounded answer when `GROQ_API_KEY` is configured.
5. Displays source names with the answer.

**No-key demo mode:** The application also runs without a Groq key. It performs the same local retrieval and shows the matched source passages, but it does not generate a new LLM answer. Add a Groq key to enable AI-generated explanations.

### 3. Linear Algebra Knowledge Graph

The NetworkX graph includes:

- **Document nodes** — uploaded textbooks, notes, and the starter course notes.
- **Concept nodes** — matrices, determinants, Gaussian elimination, rank, vector spaces, basis, linear transformations, eigenvalues, orthogonality, least squares, and related topics.
- **Document–chunk relationships** — `CONTAINS`.
- **Chunk–concept relationships** — `MENTIONS`.
- **Concept relationships** — co-occurrence and curated mathematical relationships such as `REQUIRES`, `PRODUCES`, `DEFINES`, and `REPRESENTED_BY`.

This graph is used both for exploration in the Knowledge Graph page and for graph-enhanced retrieval in the tutor.

### 4. Document Library

Upload course material in PDF, DOCX, PPTX, TXT, and Markdown formats. The ingestion pipeline extracts text, splits it into overlapping chunks, creates local embeddings, extracts concepts, and updates the knowledge graph. Documents are persisted under `data/raw/uploads/` and indexed on the local machine.

### 5. Browser interface

The FastAPI backend serves the frontend directly. Open the single local address:

`http://127.0.0.1:8000`

The interface contains four workspaces: AI Tutor, Matrix Lab, Knowledge Graph, and Document Library.

## Interface preview

These are static visual previews of the interface. The actual application is served by FastAPI at the local URL shown above.

![MatrixMind AI Tutor](previews/AI_Tutor_UI_Preview.png)

![MatrixMind Matrix Lab](previews/Matrix_Lab_UI_Preview.png)

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, SVG |
| API | Python, FastAPI, Uvicorn |
| Matrix computation | NumPy |
| Vector retrieval | Local feature hashing + NumPy cosine similarity |
| Knowledge graph | NetworkX |
| Document ingestion | PyMuPDF, python-docx, python-pptx |
| AI generation | Groq API (optional) |
| Conversation history | SQLite |

The local embedding engine does not download a transformer model. This makes the first installation smaller and permits offline indexing/retrieval. Groq is used only for generated answers and requires an API key.

## Project structure

```text
MatrixMind/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── algebra.py
│   │   ├── chat.py
│   │   ├── documents.py
│   │   ├── graph.py
│   │   ├── health.py
│   │   ├── history.py
│   │   └── v3.py
│   └── static/
│       └── index.html
├── data/
│   └── raw/
│       └── linear_algebra_fundamentals.txt
├── scripts/
├── src/
│   ├── embeddings/
│   ├── ingestion/
│   ├── knowledge_graph/
│   ├── llm/
│   ├── retrieval/
│   └── utils/
├── tests/
├── .env.example
├── requirements.txt
└── run.ps1
```

## Run on Windows PowerShell

### Prerequisites

- Windows 10/11
- Python 3.11 recommended (Python 3.12/3.13 should also work with the current dependencies)
- Internet access for the first `pip install`, and for Groq-generated answers if a key is configured

### Option A — automatic start script

Open PowerShell in the `MatrixMind` folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

The script creates `.venv`, installs dependencies, creates `.env` from `.env.example` when needed, and starts the server.

### Option B — run each command manually

```powershell
# 1. Move into the project folder
cd "C:\path\to\MatrixMind"

# 2. Create a virtual environment
py -3.11 -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks script activation, run this once in the current window:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Create your environment file
Copy-Item .env.example .env

# 6. Optional: add your Groq API key to .env
notepad .env

# 7. Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then visit **http://127.0.0.1:8000**. The starter linear-algebra notes are automatically indexed on the first launch. Keep the terminal running while using the application. Press `Ctrl + C` to stop the server.

### Groq API key (optional, recommended)

1. Create an API key from the Groq Console: <https://console.groq.com>.
2. Open the project's `.env` file.
3. Replace the blank value with your key:

```dotenv
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=openai/gpt-oss-120b
```

4. Save the file and restart Uvicorn.

Never upload your `.env` file or commit it to GitHub. Without a key, the application still supports matrix calculations, graph navigation, document ingestion, and local retrieval with matched passages.

## Useful API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | API/model status |
| POST | `/api/v1/algebra/calculate` | Matrix calculations |
| POST | `/api/v1/chat/query` | RAG tutor question |
| GET | `/api/v1/graph/nodes` | Knowledge graph nodes and links |
| GET | `/api/v1/graph/node/{node_id}` | Graph node details |
| GET | `/api/v1/documents` | Indexed documents |
| POST | `/api/v1/documents/upload` | Upload and index notes |
| GET | `/api/v1/docs` | Interactive API documentation |

## Example calculation API request

```powershell
$body = @{
  operation = "multiply"
  matrix_a = @(@(1, 2), @(3, 4))
  matrix_b = @(@(5, 6), @(7, 8))
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/algebra/calculate" -Method Post -ContentType "application/json" -Body $body
```

Expected matrix product:

```text
[[19, 22],
 [43, 50]]
```

## Testing

```powershell
pip install pytest httpx
python -m pytest -q
```

## Academic project statement

MatrixMind demonstrates the integration of numerical linear algebra, semantic retrieval, a subject knowledge graph, document processing, and optional LLM-based explanation in one academic learning application. It is more than a matrix calculator: its graph stores relationships between mathematical concepts, while RAG retrieves course material to ground answers in the student's own notes.

## Important limitations

- The local feature-hashing embedder is compact and offline-friendly; it is not as semantically rich as a large transformer embedding model. Domain term extraction and lexical scoring help improve linear-algebra retrieval.
- The no-key tutor mode is an extractive retrieval preview, not a generative AI model. Configure Groq to get generated explanations.
- The matrix workbench limits each matrix dimension to 12 for browser-demo safety and responsiveness.
- The included starter notes provide foundational learning content. Upload your own syllabus, textbook extracts, and lecture notes to extend the course knowledge base..

