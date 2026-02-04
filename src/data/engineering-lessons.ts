import { Lesson } from './lessons';

export const engineeringLessons: Lesson[] = [
  {
    id: 20,
    title: "Fine-Tuning LLMs",
    description: "Learn when and how to fine-tune language models for specific tasks",
    duration: "90 min",
    difficulty: "Advanced",
    content: `
      <h2>When to Fine-Tune vs Prompt</h2>
      <p>Fine-tuning adapts a pre-trained model to your specific task by training on your data.</p>

      <h3>Use Prompting When:</h3>
      <ul>
        <li>Task can be described in natural language</li>
        <li>You have few examples (&lt;100)</li>
        <li>Quick iteration is needed</li>
        <li>Task requirements change frequently</li>
      </ul>

      <h3>Fine-Tune When:</h3>
      <ul>
        <li>Consistent output format is critical</li>
        <li>You have lots of examples (1000+)</li>
        <li>Latency/cost reduction is important</li>
        <li>Domain-specific terminology/style needed</li>
        <li>Prompting consistently fails</li>
      </ul>

      <h3>Types of Fine-Tuning</h3>

      <h4>1. Full Fine-Tuning</h4>
      <p>Update all model parameters. Expensive but most powerful.</p>

      <h4>2. LoRA (Low-Rank Adaptation)</h4>
      <p>Only train small adapter matrices. 10-100x cheaper.</p>
      <div class="code-block">
# LoRA concept:
# Instead of updating W (d × k), train two small matrices:
# A (d × r) and B (r × k) where r << d, k

# Original: h = Wx
# LoRA: h = Wx + BAx

# Parameters reduced from d*k to r*(d+k)
# For d=k=4096, r=8: 33M → 65K parameters
      </div>

      <h4>3. QLoRA</h4>
      <p>LoRA with quantized base model. Train on consumer GPUs.</p>

      <h3>Fine-Tuning with OpenAI</h3>
      <div class="code-block">
# 1. Prepare data (JSONL format)
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}

# 2. Upload training file
from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 3. Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

# 4. Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo:my-org::abc123",
    messages=[{"role": "user", "content": "Hello"}]
)
      </div>

      <h3>Fine-Tuning Open Models with LoRA</h3>
      <div class="code-block">
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # Quantization
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Rank
    lora_alpha=32,          # Scaling
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Which layers to adapt
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# Training
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save adapter weights
model.save_pretrained("./lora-adapter")
      </div>

      <h3>Data Preparation Best Practices</h3>
      <ul>
        <li><strong>Quality over quantity</strong> - 1000 high-quality examples > 10000 poor ones</li>
        <li><strong>Diverse examples</strong> - Cover edge cases and variations</li>
        <li><strong>Consistent format</strong> - Same structure across all examples</li>
        <li><strong>Clean data</strong> - Remove duplicates, fix errors</li>
        <li><strong>Balanced classes</strong> - If classification, balance the labels</li>
      </ul>

      <h3>Evaluation</h3>
      <div class="code-block">
# Hold out test set (never train on this)
train_data, test_data = split_data(data, test_size=0.1)

# Metrics depend on task:
# - Classification: accuracy, F1, precision, recall
# - Generation: BLEU, ROUGE, human evaluation
# - Custom: task-specific metrics

# Always compare to baseline (prompting)
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Try prompting first - fine-tune only when needed</p>
        <p>2. LoRA makes fine-tuning accessible on consumer hardware</p>
        <p>3. Data quality matters more than quantity</p>
        <p>4. Always evaluate against a baseline and test set</p>
      </div>
    `,
    questions: [
      {
        id: "13-1",
        type: "mcq",
        question: "What is LoRA and why is it useful?",
        options: [
          "A new LLM architecture",
          "A technique that trains small adapter matrices instead of all parameters, making fine-tuning much cheaper",
          "A method for data augmentation",
          "A prompt engineering technique"
        ],
        correctAnswer: "A technique that trains small adapter matrices instead of all parameters, making fine-tuning much cheaper",
        explanation: "LoRA (Low-Rank Adaptation) trains two small matrices A and B instead of updating the full weight matrix W. This reduces trainable parameters by 10-100x while maintaining quality."
      },
      {
        id: "13-2",
        type: "descriptive",
        question: "When should you fine-tune a model vs use prompting? List criteria for each approach.",
        keywords: ["prompt", "examples", "cost", "latency", "consistent", "format", "domain", "quick", "iteration", "data"],
        explanation: "Prompting: few examples, quick iteration needed, changing requirements, task describable in text. Fine-tuning: many examples (1000+), consistent output format critical, domain-specific needs, cost/latency reduction needed, prompting fails."
      }
    ]
  },
  {
    id: 21,
    title: "RAG Systems Deep Dive",
    description: "Build production-ready Retrieval-Augmented Generation systems",
    duration: "100 min",
    difficulty: "Advanced",
    content: `
      <h2>RAG Architecture in Depth</h2>
      <p>RAG extends LLMs with external knowledge, reducing hallucinations and enabling domain-specific answers.</p>

      <h3>Complete RAG Pipeline</h3>
      <div class="code-block">
INDEXING PIPELINE:
Documents → Chunking → Embedding → Vector Store

QUERY PIPELINE:
Query → Embedding → Retrieval → Reranking → Generation
      </div>

      <h3>1. Document Processing</h3>

      <h4>Chunking Strategies</h4>
      <div class="code-block">
# Fixed-size chunking
def fixed_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Semantic chunking (by paragraph/section)
def semantic_chunk(text):
    # Split by double newlines (paragraphs)
    paragraphs = text.split("\\n\\n")
    return [p.strip() for p in paragraphs if p.strip()]

# Recursive chunking (LangChain style)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)
      </div>

      <h4>Chunk Size Trade-offs</h4>
      <ul>
        <li><strong>Small chunks (100-300 tokens)</strong> - More precise retrieval, but may lack context</li>
        <li><strong>Large chunks (500-1000 tokens)</strong> - More context, but may include irrelevant info</li>
        <li><strong>Sweet spot</strong> - Usually 300-500 tokens with 50-100 overlap</li>
      </ul>

      <h3>2. Embedding Models</h3>
      <div class="code-block">
# OpenAI Embeddings
from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Sentence Transformers (open source)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Goodbye world"])

# Comparison
# OpenAI text-embedding-3-small: 1536 dims, excellent quality
# OpenAI text-embedding-3-large: 3072 dims, best quality
# all-MiniLM-L6-v2: 384 dims, fast, good for simple tasks
# all-mpnet-base-v2: 768 dims, better quality, slower
      </div>

      <h3>3. Vector Databases</h3>
      <div class="code-block">
# Pinecone (managed)
import pinecone

pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Upsert
index.upsert(vectors=[
    {"id": "doc1", "values": embedding, "metadata": {"source": "file.pdf"}}
])

# Query
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

# Chroma (local/lightweight)
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=["doc1 text", "doc2 text"],
    metadatas=[{"source": "a"}, {"source": "b"}],
    ids=["id1", "id2"]
)

results = collection.query(query_texts=["search query"], n_results=5)

# pgvector (PostgreSQL)
# CREATE EXTENSION vector;
# CREATE TABLE items (id serial, embedding vector(1536));
# SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
      </div>

      <h3>4. Retrieval Strategies</h3>

      <h4>Basic Similarity Search</h4>
      <div class="code-block">
# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
      </div>

      <h4>Hybrid Search</h4>
      <p>Combine semantic (vector) + keyword (BM25) search:</p>
      <div class="code-block">
from rank_bm25 import BM25Okapi

# BM25 for keyword matching
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
bm25_scores = bm25.get_scores(query.split())

# Combine with vector scores
def hybrid_search(query, documents, alpha=0.5):
    # Vector search
    query_emb = get_embedding(query)
    vector_scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]

    # BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Normalize and combine
    vector_norm = normalize(vector_scores)
    bm25_norm = normalize(bm25_scores)

    final_scores = alpha * vector_norm + (1 - alpha) * bm25_norm
    return sorted(zip(documents, final_scores), key=lambda x: x[1], reverse=True)
      </div>

      <h4>Reranking</h4>
      <div class="code-block">
# Cross-encoder reranking (more accurate but slower)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get initial candidates with bi-encoder (fast)
candidates = vector_search(query, top_k=20)

# Rerank with cross-encoder (accurate)
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
      </div>

      <h3>5. Generation with Context</h3>
      <div class="code-block">
def generate_answer(query, retrieved_docs):
    context = "\\n\\n".join([doc.content for doc in retrieved_docs])

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Instructions:
- Only use information from the context
- If the answer isn't in the context, say "I don't have information about that"
- Cite which document the information came from

Answer:"""

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
      </div>

      <h3>6. Advanced RAG Patterns</h3>

      <h4>Query Transformation</h4>
      <div class="code-block">
# HyDE: Hypothetical Document Embedding
def hyde_search(query):
    # Generate hypothetical answer
    hypo_doc = llm.generate(f"Write a passage that answers: {query}")
    # Embed the hypothetical document
    hypo_embedding = get_embedding(hypo_doc)
    # Search with hypothetical embedding
    return vector_search(hypo_embedding)

# Multi-query: Generate multiple query variations
def multi_query_search(query):
    variations = llm.generate(f"Generate 3 variations of: {query}")
    all_results = []
    for var in variations:
        all_results.extend(vector_search(var))
    return deduplicate(all_results)
      </div>

      <h4>Self-RAG</h4>
      <p>Model decides when to retrieve:</p>
      <div class="code-block">
def self_rag(query):
    # Ask model if retrieval is needed
    needs_retrieval = llm.generate(
        f"Do you need external information to answer: {query}? Yes/No"
    )

    if "yes" in needs_retrieval.lower():
        docs = retrieve(query)
        return generate_with_context(query, docs)
    else:
        return llm.generate(query)
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Chunk size matters - experiment with 300-500 tokens</p>
        <p>2. Hybrid search (vector + BM25) often outperforms either alone</p>
        <p>3. Reranking improves precision at retrieval cost</p>
        <p>4. Query transformation can significantly improve results</p>
        <p>5. Always include source citations in generated answers</p>
      </div>
    `,
    questions: [
      {
        id: "14-1",
        type: "mcq",
        question: "What is hybrid search in RAG?",
        options: [
          "Using two different LLMs",
          "Combining vector similarity search with keyword (BM25) search",
          "Searching across multiple databases",
          "Using both CPU and GPU"
        ],
        correctAnswer: "Combining vector similarity search with keyword (BM25) search",
        explanation: "Hybrid search combines semantic understanding (vector search) with exact keyword matching (BM25). This catches both semantically similar content and exact term matches."
      },
      {
        id: "14-2",
        type: "descriptive",
        question: "Explain the trade-offs in choosing chunk size for RAG systems.",
        keywords: ["small", "large", "context", "precise", "retrieval", "relevant", "overlap", "tokens", "information"],
        explanation: "Small chunks (100-300 tokens) give precise retrieval but may lack context. Large chunks (500-1000 tokens) provide more context but may include irrelevant info. Sweet spot is usually 300-500 tokens with 50-100 overlap to maintain context across boundaries."
      }
    ]
  },
  {
    id: 22,
    title: "Building & Deploying AI CLI Tools",
    description: "Create command-line AI tools and deploy them for users",
    duration: "80 min",
    difficulty: "Advanced",
    content: `
      <h2>Building AI-Powered CLI Applications</h2>
      <p>CLI tools are powerful ways to deliver AI capabilities. Users can integrate them into workflows, scripts, and automation.</p>

      <h3>CLI Framework Options</h3>
      <ul>
        <li><strong>Python: Click/Typer</strong> - Most popular, great for AI tools</li>
        <li><strong>Node.js: Commander/Yargs</strong> - Good for JS ecosystem</li>
        <li><strong>Go: Cobra</strong> - Fast, single binary distribution</li>
        <li><strong>Rust: Clap</strong> - Extremely fast, safe</li>
      </ul>

      <h3>Building with Python + Typer</h3>

      <h4>Project Structure</h4>
      <div class="code-block">
my-ai-cli/
├── pyproject.toml
├── src/
│   └── my_ai_cli/
│       ├── __init__.py
│       ├── main.py
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   ├── analyze.py
│       │   └── generate.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           └── api.py
└── tests/
      </div>

      <h4>Main CLI Entry Point</h4>
      <div class="code-block">
# src/my_ai_cli/main.py
import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer(
    name="ai-tool",
    help="AI-powered CLI tool for various tasks",
    add_completion=True
)
console = Console()

@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send to AI"),
    model: str = typer.Option("claude-3-opus", "--model", "-m", help="Model to use"),
    system: str = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output")
):
    """Chat with an AI model."""
    from .utils.api import get_response

    with console.status("[bold green]Thinking..."):
        response = get_response(message, model, system, stream)

    if stream:
        for chunk in response:
            console.print(chunk, end="")
    else:
        console.print(Markdown(response))

@app.command()
def analyze(
    file: str = typer.Argument(..., help="File to analyze"),
    task: str = typer.Option("summarize", "--task", "-t",
                             help="Analysis task: summarize, review, explain")
):
    """Analyze a file with AI."""
    import os

    if not os.path.exists(file):
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    with open(file, 'r') as f:
        content = f.read()

    prompts = {
        "summarize": f"Summarize this file:\\n\\n{content}",
        "review": f"Review this code for issues:\\n\\n{content}",
        "explain": f"Explain what this code does:\\n\\n{content}"
    }

    from .utils.api import get_response
    response = get_response(prompts[task])
    console.print(Markdown(response))

@app.command()
def config(
    api_key: str = typer.Option(None, "--api-key", help="Set API key"),
    show: bool = typer.Option(False, "--show", help="Show current config")
):
    """Configure the CLI tool."""
    from .utils.config import Config

    cfg = Config()

    if show:
        console.print(cfg.display())
        return

    if api_key:
        cfg.set("api_key", api_key)
        console.print("[green]API key saved![/green]")

if __name__ == "__main__":
    app()
      </div>

      <h4>Configuration Management</h4>
      <div class="code-block">
# src/my_ai_cli/utils/config.py
import os
import json
from pathlib import Path

class Config:
    def __init__(self):
        self.config_dir = Path.home() / ".my-ai-cli"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self.data = self._load()

    def _load(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def _save(self):
        self.config_file.write_text(json.dumps(self.data, indent=2))

    def get(self, key, default=None):
        # Check environment variable first
        env_key = f"MY_AI_CLI_{key.upper()}"
        return os.environ.get(env_key) or self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self._save()
      </div>

      <h4>API Integration</h4>
      <div class="code-block">
# src/my_ai_cli/utils/api.py
import anthropic
from .config import Config

def get_client():
    config = Config()
    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("API key not set. Run: ai-tool config --api-key YOUR_KEY")
    return anthropic.Anthropic(api_key=api_key)

def get_response(message, model="claude-3-opus-20240229", system=None, stream=False):
    client = get_client()

    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": message}]
    }
    if system:
        kwargs["system"] = system

    if stream:
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    else:
        response = client.messages.create(**kwargs)
        return response.content[0].text
      </div>

      <h3>Interactive Mode</h3>
      <div class="code-block">
@app.command()
def interactive():
    """Start interactive chat session."""
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory

    console.print("[bold]Interactive AI Chat[/bold]")
    console.print("Type 'exit' to quit, 'clear' to reset\\n")

    history = FileHistory(str(Path.home() / ".my-ai-cli" / "history"))
    messages = []

    while True:
        try:
            user_input = prompt("You: ", history=history)
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'clear':
            messages = []
            console.print("[dim]Conversation cleared[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})

        console.print("AI: ", end="")
        response_text = ""
        for chunk in get_streaming_response(messages):
            console.print(chunk, end="")
            response_text += chunk
        console.print()

        messages.append({"role": "assistant", "content": response_text})
      </div>

      <h3>Packaging & Distribution</h3>

      <h4>pyproject.toml</h4>
      <div class="code-block">
[project]
name = "my-ai-cli"
version = "1.0.0"
description = "AI-powered CLI tool"
requires-python = ">=3.9"
dependencies = [
    "typer[all]>=0.9.0",
    "anthropic>=0.18.0",
    "rich>=13.0.0",
    "prompt-toolkit>=3.0.0",
]

[project.scripts]
ai-tool = "my_ai_cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
      </div>

      <h4>Publishing to PyPI</h4>
      <div class="code-block">
# Build
pip install build
python -m build

# Upload to PyPI
pip install twine
twine upload dist/*

# Users can now install with:
pip install my-ai-cli
      </div>

      <h4>Creating Standalone Executables</h4>
      <div class="code-block">
# Using PyInstaller
pip install pyinstaller
pyinstaller --onefile src/my_ai_cli/main.py --name ai-tool

# Using Nuitka (faster executables)
pip install nuitka
nuitka --standalone --onefile src/my_ai_cli/main.py
      </div>

      <h3>Distribution Strategies</h3>
      <ul>
        <li><strong>PyPI</strong> - pip install my-ai-cli</li>
        <li><strong>Homebrew</strong> - brew install my-ai-cli</li>
        <li><strong>npm</strong> - npm install -g my-ai-cli (for Node)</li>
        <li><strong>GitHub Releases</strong> - Binary downloads</li>
        <li><strong>Docker</strong> - docker run my-ai-cli</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Use Typer/Click for Python CLIs - great developer experience</p>
        <p>2. Store config in ~/.your-app/ directory</p>
        <p>3. Support both env vars and config file for API keys</p>
        <p>4. Add streaming for better UX on long responses</p>
        <p>5. Publish to PyPI for easy installation</p>
      </div>
    `,
    questions: [
      {
        id: "15-1",
        type: "mcq",
        question: "What is the recommended way to handle API keys in a CLI tool?",
        options: [
          "Hardcode them in the source code",
          "Always require them as command arguments",
          "Support both environment variables and a config file",
          "Store them in a public config file"
        ],
        correctAnswer: "Support both environment variables and a config file",
        explanation: "CLI tools should check environment variables first (for CI/CD and scripts) and fall back to a config file in the user's home directory (for local development). Never hardcode secrets."
      },
      {
        id: "15-2",
        type: "descriptive",
        question: "Describe how to package and distribute a Python CLI tool so users can install it with pip.",
        keywords: ["pyproject.toml", "scripts", "entry point", "build", "PyPI", "twine", "upload", "pip install"],
        explanation: "Create pyproject.toml with [project.scripts] defining the CLI entry point. Build with 'python -m build', then upload to PyPI with 'twine upload dist/*'. Users can then 'pip install your-package'."
      }
    ]
  },
  {
    id: 23,
    title: "Prompt Engineering Mastery",
    description: "Advanced techniques for crafting effective prompts",
    duration: "70 min",
    difficulty: "Intermediate",
    content: `
      <h2>The Art & Science of Prompt Engineering</h2>
      <p>Prompts are the interface between humans and LLMs. Great prompts unlock the model's capabilities.</p>

      <h3>Prompt Structure</h3>
      <div class="code-block">
[SYSTEM PROMPT - Sets behavior, personality, constraints]
[CONTEXT - Background information, documents]
[EXAMPLES - Few-shot demonstrations]
[INSTRUCTION - What to do]
[INPUT - User's specific request]
[OUTPUT FORMAT - How to structure the response]
      </div>

      <h3>Core Techniques</h3>

      <h4>1. Be Specific and Explicit</h4>
      <div class="code-block">
# Bad
Summarize this article.

# Good
Summarize this article in exactly 3 bullet points.
Each bullet should be one sentence, max 20 words.
Focus on the main argument, key evidence, and conclusion.
      </div>

      <h4>2. Provide Context</h4>
      <div class="code-block">
# Bad
Write a product description.

# Good
You are a copywriter for a luxury watch brand.
Target audience: affluent professionals aged 35-55.
Tone: sophisticated, exclusive, not flashy.

Write a 100-word product description for our new titanium watch.
      </div>

      <h4>3. Few-Shot Learning</h4>
      <div class="code-block">
Classify the sentiment of these reviews:

Review: "The food was amazing and service was great!"
Sentiment: Positive

Review: "Waited 2 hours, food was cold"
Sentiment: Negative

Review: "It was okay, nothing special"
Sentiment: Neutral

Review: "Best experience ever, will definitely return!"
Sentiment:
      </div>

      <h4>4. Chain-of-Thought (CoT)</h4>
      <div class="code-block">
# Bad
What is 23 * 47?

# Good
What is 23 * 47?
Think step by step and show your work.

# Even better (with example)
Q: What is 15 * 23?
A: Let me solve this step by step:
   15 * 23 = 15 * 20 + 15 * 3
   = 300 + 45
   = 345

Q: What is 23 * 47?
A: Let me solve this step by step:
      </div>

      <h4>5. Role Prompting</h4>
      <div class="code-block">
You are an expert security researcher with 20 years of experience.
You specialize in finding vulnerabilities in web applications.
When reviewing code, you:
- Think adversarially
- Consider edge cases
- Explain vulnerabilities clearly
- Suggest specific fixes

Review this code for security issues:
[code here]
      </div>

      <h4>6. Output Formatting</h4>
      <div class="code-block">
Analyze this data and respond in this exact JSON format:
{
  "summary": "one sentence summary",
  "key_points": ["point 1", "point 2", "point 3"],
  "sentiment": "positive|negative|neutral",
  "confidence": 0.0-1.0
}

Do not include any text outside the JSON.
      </div>

      <h3>Advanced Techniques</h3>

      <h4>Self-Consistency</h4>
      <p>Generate multiple responses and take the majority answer:</p>
      <div class="code-block">
# Ask the same question 5 times with temperature > 0
# Take the most common answer
# Works well for reasoning tasks
      </div>

      <h4>Tree of Thoughts</h4>
      <div class="code-block">
Solve this problem using tree of thoughts:

1. Generate 3 different initial approaches
2. For each approach, think 2 steps ahead
3. Evaluate which branch is most promising
4. Continue with the best branch
5. If stuck, backtrack and try another branch

Problem: [complex problem here]
      </div>

      <h4>ReAct Prompting</h4>
      <div class="code-block">
Answer the following question using this format:

Thought: [reason about what to do]
Action: [action to take]
Observation: [result of action]
... (repeat as needed)
Thought: I now know the answer
Answer: [final answer]

Question: What was the GDP of France in the year the Eiffel Tower was built?
      </div>

      <h3>Prompt Templates</h3>

      <h4>Code Review Template</h4>
      <div class="code-block">
Review this {language} code:

\`\`\`{language}
{code}
\`\`\`

Evaluate:
1. **Correctness**: Does it work as intended?
2. **Security**: Any vulnerabilities?
3. **Performance**: Any inefficiencies?
4. **Readability**: Is it clear and maintainable?
5. **Best Practices**: Does it follow {language} conventions?

For each issue found, provide:
- Severity (Critical/High/Medium/Low)
- Line number(s)
- Description
- Suggested fix with code
      </div>

      <h4>Writing Assistant Template</h4>
      <div class="code-block">
You are an expert editor. Improve this text while preserving the author's voice.

Original text:
{text}

Improve for:
- [ ] Clarity
- [ ] Conciseness
- [ ] Grammar
- [ ] Flow

Output format:
1. Edited text
2. Summary of changes made
3. Suggestions for further improvement
      </div>

      <h3>Common Mistakes</h3>
      <ul>
        <li><strong>Too vague</strong> - "Make it better" vs specific criteria</li>
        <li><strong>Missing context</strong> - Assuming the model knows your situation</li>
        <li><strong>Overloading</strong> - Too many instructions at once</li>
        <li><strong>Wrong format</strong> - Asking for JSON but allowing prose</li>
        <li><strong>No examples</strong> - When the task is ambiguous</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Be specific about what you want and how you want it</p>
        <p>2. Provide relevant context and examples</p>
        <p>3. Use Chain-of-Thought for complex reasoning</p>
        <p>4. Specify output format explicitly</p>
        <p>5. Iterate and refine based on results</p>
      </div>
    `,
    questions: [
      {
        id: "16-1",
        type: "mcq",
        question: "What is Chain-of-Thought (CoT) prompting?",
        options: [
          "Connecting multiple AI models together",
          "Asking the model to show its reasoning step by step before giving the final answer",
          "A method for training models faster",
          "Using multiple prompts in sequence"
        ],
        correctAnswer: "Asking the model to show its reasoning step by step before giving the final answer",
        explanation: "Chain-of-Thought prompting encourages the model to break down complex problems into steps, showing its reasoning process. This often improves accuracy on math, logic, and multi-step reasoning tasks."
      },
      {
        id: "16-2",
        type: "descriptive",
        question: "What are the key components of an effective prompt structure?",
        keywords: ["system", "context", "examples", "instruction", "format", "output", "specific", "role"],
        explanation: "Effective prompts include: System prompt (behavior/role), Context (background info), Examples (few-shot demonstrations), Clear instruction (what to do), Input (user's request), and Output format (how to structure response)."
      }
    ]
  }
];
