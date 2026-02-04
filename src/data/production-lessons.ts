import { Lesson } from './lessons';

export const productionLessons: Lesson[] = [
  {
    id: 24,
    title: "MLOps & AI Infrastructure",
    description: "Deploy, monitor, and scale AI systems in production",
    duration: "90 min",
    difficulty: "Advanced",
    content: `
      <h2>Production AI Infrastructure</h2>
      <p>Taking AI from prototype to production requires proper infrastructure, monitoring, and operational practices.</p>

      <h3>MLOps Lifecycle</h3>
      <div class="code-block">
Data → Train → Evaluate → Deploy → Monitor → Retrain
  ↑_______________________________________|
      </div>

      <h3>Deployment Patterns</h3>

      <h4>1. API Service</h4>
      <div class="code-block">
# FastAPI example
from fastapi import FastAPI
import anthropic

app = FastAPI()
client = anthropic.Anthropic()

@app.post("/generate")
async def generate(prompt: str):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"response": response.content[0].text}

# Deploy with: uvicorn main:app --host 0.0.0.0 --port 8000
      </div>

      <h4>2. Serverless Functions</h4>
      <div class="code-block">
# Vercel/AWS Lambda
export async function POST(request: Request) {
  const { prompt } = await request.json();

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'content-type': 'application/json',
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-3-haiku-20240307',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }]
    })
  });

  return Response.json(await response.json());
}
      </div>

      <h4>3. Batch Processing</h4>
      <div class="code-block">
# Process large datasets asynchronously
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def process_batch(items, concurrency=10):
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(item):
        async with semaphore:
            response = await client.messages.create(...)
            return response

    tasks = [process_one(item) for item in items]
    return await asyncio.gather(*tasks)
      </div>

      <h3>Scaling Strategies</h3>

      <h4>Rate Limiting</h4>
      <div class="code-block">
from ratelimit import limits, sleep_and_retry

# 60 requests per minute
@sleep_and_retry
@limits(calls=60, period=60)
def call_api(prompt):
    return client.messages.create(...)
      </div>

      <h4>Caching</h4>
      <div class="code-block">
import redis
import hashlib
import json

cache = redis.Redis()

def cached_generate(prompt, ttl=3600):
    # Create cache key from prompt
    key = hashlib.md5(prompt.encode()).hexdigest()

    # Check cache
    cached = cache.get(key)
    if cached:
        return json.loads(cached)

    # Generate and cache
    response = generate(prompt)
    cache.setex(key, ttl, json.dumps(response))
    return response
      </div>

      <h4>Load Balancing</h4>
      <div class="code-block">
# Multiple API keys for higher throughput
import itertools

api_keys = ["key1", "key2", "key3"]
key_cycle = itertools.cycle(api_keys)

def get_client():
    key = next(key_cycle)
    return anthropic.Anthropic(api_key=key)
      </div>

      <h3>Monitoring & Observability</h3>

      <h4>Logging</h4>
      <div class="code-block">
import logging
import json
from datetime import datetime

logger = logging.getLogger("ai_service")

def log_request(prompt, response, latency):
    logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "latency_ms": latency,
        "model": "claude-3-opus",
        "tokens_used": response.usage.total_tokens
    }))
      </div>

      <h4>Metrics</h4>
      <div class="code-block">
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
requests_total = Counter('ai_requests_total', 'Total AI requests', ['model', 'status'])
latency_seconds = Histogram('ai_latency_seconds', 'AI request latency')
tokens_used = Counter('ai_tokens_total', 'Total tokens used', ['model'])

# Use in code
@latency_seconds.time()
def generate(prompt):
    response = client.messages.create(...)
    requests_total.labels(model='claude', status='success').inc()
    tokens_used.labels(model='claude').inc(response.usage.total_tokens)
    return response
      </div>

      <h4>Alerting</h4>
      <div class="code-block">
# Example alert rules (Prometheus/Grafana)
- alert: HighLatency
  expr: histogram_quantile(0.95, ai_latency_seconds) > 5
  for: 5m
  annotations:
    summary: "AI latency is high"

- alert: HighErrorRate
  expr: rate(ai_requests_total{status="error"}[5m]) > 0.1
  annotations:
    summary: "AI error rate above 10%"
      </div>

      <h3>Cost Optimization</h3>
      <ul>
        <li><strong>Model selection</strong> - Use Haiku for simple tasks, Opus for complex</li>
        <li><strong>Prompt optimization</strong> - Shorter prompts = fewer tokens</li>
        <li><strong>Caching</strong> - Don't regenerate identical requests</li>
        <li><strong>Batching</strong> - Process in bulk when possible</li>
        <li><strong>Token budgets</strong> - Set max_tokens appropriately</li>
      </ul>

      <div class="code-block">
# Smart model routing
def route_to_model(prompt, complexity_score):
    if complexity_score < 0.3:
        return "claude-3-haiku-20240307"  # $0.25/1M tokens
    elif complexity_score < 0.7:
        return "claude-3-sonnet-20240229"  # $3/1M tokens
    else:
        return "claude-3-opus-20240229"    # $15/1M tokens
      </div>

      <h3>Security</h3>
      <ul>
        <li><strong>Input validation</strong> - Sanitize user inputs</li>
        <li><strong>Output filtering</strong> - Check for harmful content</li>
        <li><strong>Rate limiting</strong> - Prevent abuse</li>
        <li><strong>API key security</strong> - Never expose in frontend</li>
        <li><strong>Audit logging</strong> - Track all requests</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Choose deployment pattern based on use case (API, serverless, batch)</p>
        <p>2. Implement caching and rate limiting from day one</p>
        <p>3. Monitor latency, errors, and costs continuously</p>
        <p>4. Route to cheaper models when possible</p>
        <p>5. Security: validate inputs, filter outputs, protect keys</p>
      </div>
    `,
    questions: [
      {
        id: "17-1",
        type: "mcq",
        question: "What is a key benefit of caching AI responses?",
        options: [
          "It makes the model smarter",
          "It reduces costs and latency for repeated queries",
          "It improves model accuracy",
          "It increases the context window"
        ],
        correctAnswer: "It reduces costs and latency for repeated queries",
        explanation: "Caching stores responses so identical requests don't need to call the API again. This reduces both API costs (fewer calls) and latency (instant cache hits)."
      },
      {
        id: "17-2",
        type: "descriptive",
        question: "Describe a strategy for optimizing AI API costs in production.",
        keywords: ["model", "routing", "cache", "haiku", "opus", "tokens", "prompt", "batch", "cheaper"],
        explanation: "Cost optimization strategies: 1) Route simple tasks to cheaper models (Haiku), complex to expensive (Opus), 2) Cache identical requests, 3) Optimize prompts to reduce tokens, 4) Set appropriate max_tokens, 5) Batch process when possible."
      }
    ]
  },
  {
    id: 25,
    title: "Embeddings & Vector Search",
    description: "Deep dive into embeddings, similarity search, and vector databases",
    duration: "75 min",
    difficulty: "Advanced",
    content: `
      <h2>Understanding Embeddings</h2>
      <p>Embeddings are dense vector representations that capture semantic meaning. They're the foundation of modern search, RAG, and recommendation systems.</p>

      <h3>What Are Embeddings?</h3>
      <div class="code-block">
"Hello world" → [0.023, -0.156, 0.089, ..., 0.234]  # 1536 dimensions

Properties:
- Similar meanings → similar vectors
- king - man + woman ≈ queen
- Enables semantic search (meaning, not just keywords)
      </div>

      <h3>Embedding Models</h3>
      <table>
        <tr><th>Model</th><th>Dimensions</th><th>Best For</th></tr>
        <tr><td>OpenAI text-embedding-3-small</td><td>1536</td><td>General purpose, cost-effective</td></tr>
        <tr><td>OpenAI text-embedding-3-large</td><td>3072</td><td>Highest quality</td></tr>
        <tr><td>Cohere embed-v3</td><td>1024</td><td>Multilingual</td></tr>
        <tr><td>all-MiniLM-L6-v2</td><td>384</td><td>Fast, local, free</td></tr>
        <tr><td>all-mpnet-base-v2</td><td>768</td><td>Better quality, local</td></tr>
      </table>

      <h3>Creating Embeddings</h3>
      <div class="code-block">
# OpenAI
from openai import OpenAI
client = OpenAI()

def embed_openai(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# Sentence Transformers (local)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Hi there"])

# Batch for efficiency
def embed_batch(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embed_openai(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings
      </div>

      <h3>Similarity Metrics</h3>
      <div class="code-block">
import numpy as np

# Cosine Similarity (most common)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Range: -1 to 1 (1 = identical direction)

# Euclidean Distance
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
# Range: 0 to ∞ (0 = identical)

# Dot Product (when vectors are normalized)
def dot_product(a, b):
    return np.dot(a, b)
# Range: -1 to 1 (for unit vectors)
      </div>

      <h3>Vector Database Deep Dive</h3>

      <h4>Indexing Algorithms</h4>
      <ul>
        <li><strong>Flat (Brute Force)</strong> - Exact, but O(n) search</li>
        <li><strong>IVF (Inverted File)</strong> - Clusters vectors, searches nearby clusters</li>
        <li><strong>HNSW (Hierarchical Navigable Small World)</strong> - Graph-based, very fast</li>
        <li><strong>LSH (Locality Sensitive Hashing)</strong> - Hash-based approximate search</li>
      </ul>

      <div class="code-block">
# FAISS example (Facebook AI Similarity Search)
import faiss
import numpy as np

# Create index
dimension = 1536
index = faiss.IndexFlatL2(dimension)  # Exact search

# Or use HNSW for speed
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter

# Add vectors
vectors = np.array(embeddings).astype('float32')
index.add(vectors)

# Search
query = np.array([query_embedding]).astype('float32')
distances, indices = index.search(query, k=5)
      </div>

      <h4>Metadata Filtering</h4>
      <div class="code-block">
# Pinecone with metadata
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": embedding,
        "metadata": {
            "source": "blog",
            "date": "2024-01-15",
            "author": "John"
        }
    }
])

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "source": {"$eq": "blog"},
        "date": {"$gte": "2024-01-01"}
    }
)
      </div>

      <h3>Building a Search System</h3>
      <div class="code-block">
class SemanticSearch:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        embeddings = self.model.encode(docs)
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            {"document": self.documents[i], "score": similarities[i]}
            for i in top_indices
        ]

# Usage
search = SemanticSearch()
search.add_documents([
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Neural networks process data in layers"
])

results = search.search("What is deep learning?")
      </div>

      <h3>Embedding Best Practices</h3>
      <ul>
        <li><strong>Normalize vectors</strong> for cosine similarity</li>
        <li><strong>Batch embedding calls</strong> for efficiency</li>
        <li><strong>Store original text</strong> alongside vectors</li>
        <li><strong>Use appropriate chunk sizes</strong> (512-1024 tokens)</li>
        <li><strong>Consider multilingual models</strong> for global content</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Embeddings convert text to vectors where similar meanings are close</p>
        <p>2. Cosine similarity is the most common metric</p>
        <p>3. HNSW provides fast approximate nearest neighbor search</p>
        <p>4. Metadata filtering enables hybrid search capabilities</p>
        <p>5. Choose embedding model based on speed vs quality tradeoffs</p>
      </div>
    `,
    questions: [
      {
        id: "18-1",
        type: "mcq",
        question: "Why is cosine similarity commonly used for comparing embeddings?",
        options: [
          "It's the fastest metric",
          "It measures the angle between vectors, which captures semantic similarity regardless of magnitude",
          "It always gives values between 0 and 1",
          "It's required by most embedding models"
        ],
        correctAnswer: "It measures the angle between vectors, which captures semantic similarity regardless of magnitude",
        explanation: "Cosine similarity measures the angle between vectors, not their magnitude. This is important because the direction of an embedding captures its semantic meaning, while magnitude can vary based on text length or other factors."
      },
      {
        id: "18-2",
        type: "descriptive",
        question: "Explain the trade-offs between exact search (Flat index) and approximate search (HNSW) for vector databases.",
        keywords: ["exact", "approximate", "speed", "accuracy", "scale", "O(n)", "fast", "memory", "index"],
        explanation: "Flat/exact search compares query to ALL vectors (O(n)) - accurate but slow at scale. HNSW builds a graph for approximate nearest neighbors - very fast (sublinear) but may miss some relevant results. For millions of vectors, approximate search is necessary."
      }
    ]
  },
  {
    id: 26,
    title: "AI System Design",
    description: "Design complete AI systems for real-world applications",
    duration: "100 min",
    difficulty: "Advanced",
    content: `
      <h2>Designing AI Systems</h2>
      <p>Building production AI systems requires thinking about architecture, scalability, reliability, and user experience.</p>

      <h3>System Design Framework</h3>
      <ol>
        <li><strong>Requirements</strong> - What problem are we solving?</li>
        <li><strong>Scale</strong> - How many users/requests?</li>
        <li><strong>Latency</strong> - What's acceptable response time?</li>
        <li><strong>Accuracy</strong> - What error rate is acceptable?</li>
        <li><strong>Cost</strong> - What's the budget?</li>
      </ol>

      <h3>Case Study 1: AI Customer Support Bot</h3>

      <h4>Requirements</h4>
      <ul>
        <li>Answer customer questions 24/7</li>
        <li>Use company knowledge base</li>
        <li>Escalate to humans when needed</li>
        <li>Support multiple languages</li>
      </ul>

      <h4>Architecture</h4>
      <div class="code-block">
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│  API Gateway │────▶│  Rate Limit  │
│   (Chat UI)  │     │   (Auth)     │     │   + Cache    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Intent     │  │   RAG       │  │  Action     │       │
│  │  Classifier │  │   Pipeline  │  │  Executor   │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└──────────────────────────────────────────────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│    LLM API   │   │ Vector Store │   │ External APIs│
│   (Claude)   │   │  (Pinecone)  │   │ (CRM, etc)   │
└──────────────┘   └──────────────┘   └──────────────┘
      </div>

      <h4>Key Components</h4>
      <div class="code-block">
# Intent Classification
intents = ["faq", "order_status", "refund", "technical", "escalate"]

def classify_intent(message):
    response = llm.generate(f"""
    Classify this customer message into one of: {intents}

    Message: {message}

    Return only the intent label.
    """)
    return response.strip()

# RAG for FAQ
def answer_faq(question):
    # Retrieve relevant docs
    docs = vector_store.search(question, top_k=3)

    # Generate answer
    context = "\\n".join([d.content for d in docs])
    answer = llm.generate(f"""
    Answer based on this context:
    {context}

    Question: {question}

    If the answer isn't in the context, say you need to escalate.
    """)
    return answer

# Escalation Logic
def should_escalate(message, history):
    signals = [
        "angry" in sentiment_analysis(message),
        "speak to human" in message.lower(),
        len(history) > 10,  # Long conversation
        confidence_score(last_answer) < 0.5
    ]
    return any(signals)
      </div>

      <h3>Case Study 2: Code Review Assistant</h3>

      <h4>Architecture</h4>
      <div class="code-block">
GitHub/GitLab Webhook
         │
         ▼
┌─────────────────┐
│   Queue (SQS)   │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           Review Pipeline                │
│  ┌───────────┐  ┌───────────┐           │
│  │  Parse    │  │  Chunk    │           │
│  │  Diff     │──▶│  Code     │           │
│  └───────────┘  └───────────┘           │
│                       │                  │
│         ┌─────────────┴──────────────┐  │
│         ▼             ▼              ▼  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐
│   │ Security │  │ Quality  │  │  Style   │
│   │ Review   │  │ Review   │  │  Review  │
│   └──────────┘  └──────────┘  └──────────┘
│         │             │              │  │
│         └─────────────┴──────────────┘  │
│                       │                  │
│                       ▼                  │
│               ┌──────────────┐          │
│               │   Aggregate  │          │
│               │   Comments   │          │
│               └──────────────┘          │
└─────────────────────────────────────────┘
         │
         ▼
    Post to PR
      </div>

      <h3>Case Study 3: Document Intelligence</h3>

      <h4>Multi-Modal Pipeline</h4>
      <div class="code-block">
Document Upload (PDF, DOCX, Images)
         │
         ▼
┌─────────────────────────────────────────┐
│          Document Processor              │
│  ┌───────────┐  ┌───────────┐           │
│  │   PDF     │  │  OCR      │           │
│  │  Parser   │  │ (Images)  │           │
│  └───────────┘  └───────────┘           │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Content Extraction               │
│  • Tables → Structured data             │
│  • Text → Chunks                        │
│  • Images → Descriptions (Vision API)   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Embedding & Storage              │
│  • Text chunks → Vector DB              │
│  • Metadata → SQL DB                    │
│  • Original → Object Storage            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Query Interface                  │
│  • Natural language questions           │
│  • Specific data extraction             │
│  • Cross-document analysis              │
└─────────────────────────────────────────┘
      </div>

      <h3>Design Patterns</h3>

      <h4>1. Graceful Degradation</h4>
      <div class="code-block">
async def generate_with_fallback(prompt):
    try:
        # Try primary model
        return await call_claude_opus(prompt)
    except RateLimitError:
        # Fall back to faster model
        return await call_claude_haiku(prompt)
    except APIError:
        # Fall back to cached response
        return get_cached_response(prompt)
    except:
        # Final fallback
        return "I'm having trouble right now. Please try again."
      </div>

      <h4>2. Human-in-the-Loop</h4>
      <div class="code-block">
def process_with_review(input):
    # AI generates initial output
    ai_output = generate(input)

    # Check confidence
    if confidence(ai_output) < THRESHOLD:
        # Queue for human review
        queue_for_review(input, ai_output)
        return {"status": "pending_review"}

    return {"status": "complete", "output": ai_output}
      </div>

      <h4>3. Feedback Loop</h4>
      <div class="code-block">
def log_interaction(input, output, feedback=None):
    db.insert({
        "input": input,
        "output": output,
        "feedback": feedback,  # thumbs up/down
        "timestamp": now()
    })

# Use feedback for:
# 1. Fine-tuning data
# 2. Prompt improvement
# 3. Quality monitoring
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Start with clear requirements: scale, latency, accuracy, cost</p>
        <p>2. Use queues for async processing of heavy workloads</p>
        <p>3. Implement graceful degradation and fallbacks</p>
        <p>4. Design for human-in-the-loop when AI confidence is low</p>
        <p>5. Collect feedback for continuous improvement</p>
      </div>
    `,
    questions: [
      {
        id: "19-1",
        type: "mcq",
        question: "What is graceful degradation in AI systems?",
        options: [
          "Slowly making the model worse over time",
          "Providing fallback responses when primary systems fail",
          "Reducing the model size",
          "Lowering accuracy requirements"
        ],
        correctAnswer: "Providing fallback responses when primary systems fail",
        explanation: "Graceful degradation means having fallback mechanisms: if the primary model fails, fall back to a simpler model; if that fails, use cache; if that fails, return a helpful error message. The system never completely fails."
      },
      {
        id: "19-2",
        type: "descriptive",
        question: "Design a high-level architecture for an AI-powered customer support system. Include key components and their interactions.",
        keywords: ["intent", "RAG", "vector", "LLM", "escalation", "knowledge base", "queue", "cache", "API"],
        explanation: "Key components: 1) API Gateway (auth, rate limiting), 2) Intent Classifier (route requests), 3) RAG Pipeline (retrieve from knowledge base), 4) LLM for response generation, 5) Vector store for semantic search, 6) Escalation logic to human agents, 7) Caching layer, 8) Feedback collection."
      }
    ]
  },
  {
    id: 27,
    title: "Capstone: Build a Complete AI Application",
    description: "Apply everything you've learned to build a production-ready AI application",
    duration: "180 min",
    difficulty: "Advanced",
    content: `
      <h2>Capstone Project: AI Research Assistant</h2>
      <p>Build a complete AI research assistant that can read papers, answer questions, and generate insights.</p>

      <h3>Project Requirements</h3>
      <ul>
        <li>Upload and process PDF research papers</li>
        <li>Ask questions about papers (RAG)</li>
        <li>Generate summaries and key findings</li>
        <li>Compare multiple papers</li>
        <li>Export notes and citations</li>
      </ul>

      <h3>Technology Stack</h3>
      <ul>
        <li><strong>Frontend</strong>: Next.js + TypeScript</li>
        <li><strong>Backend</strong>: Next.js API Routes</li>
        <li><strong>AI</strong>: Claude API</li>
        <li><strong>Vector DB</strong>: Supabase pgvector</li>
        <li><strong>Storage</strong>: Supabase Storage</li>
        <li><strong>Auth</strong>: Supabase Auth</li>
      </ul>

      <h3>Step 1: Project Setup</h3>
      <div class="code-block">
npx create-next-app@latest research-assistant --typescript --tailwind
cd research-assistant
npm install @anthropic-ai/sdk @supabase/supabase-js pdf-parse
      </div>

      <h3>Step 2: Database Schema</h3>
      <div class="code-block">
-- Enable pgvector
create extension if not exists vector;

-- Papers table
create table papers (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id),
  title text not null,
  authors text[],
  abstract text,
  file_url text,
  created_at timestamp with time zone default now()
);

-- Chunks table for RAG
create table paper_chunks (
  id uuid default gen_random_uuid() primary key,
  paper_id uuid references papers(id) on delete cascade,
  content text not null,
  embedding vector(1536),
  chunk_index integer,
  page_number integer
);

-- Similarity search function
create function match_paper_chunks(
  query_embedding vector(1536),
  paper_ids uuid[],
  match_count int default 5
) returns table (
  id uuid,
  paper_id uuid,
  content text,
  similarity float
) language plpgsql as $$
begin
  return query
  select
    pc.id,
    pc.paper_id,
    pc.content,
    1 - (pc.embedding <=> query_embedding) as similarity
  from paper_chunks pc
  where pc.paper_id = any(paper_ids)
  order by pc.embedding <=> query_embedding
  limit match_count;
end;
$$;
      </div>

      <h3>Step 3: PDF Processing</h3>
      <div class="code-block">
// lib/pdf-processor.ts
import pdf from 'pdf-parse';
import { OpenAI } from 'openai';
import { createClient } from '@supabase/supabase-js';

const openai = new OpenAI();
const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_KEY!);

export async function processPaper(file: Buffer, paperId: string) {
  // Parse PDF
  const data = await pdf(file);
  const text = data.text;

  // Extract metadata with AI
  const metadata = await extractMetadata(text.slice(0, 3000));

  // Chunk the text
  const chunks = chunkText(text, 500, 100);

  // Generate embeddings and store
  for (let i = 0; i < chunks.length; i++) {
    const embedding = await getEmbedding(chunks[i].text);

    await supabase.from('paper_chunks').insert({
      paper_id: paperId,
      content: chunks[i].text,
      embedding: embedding,
      chunk_index: i,
      page_number: chunks[i].page
    });
  }

  return metadata;
}

async function extractMetadata(text: string) {
  const response = await anthropic.messages.create({
    model: 'claude-3-haiku-20240307',
    max_tokens: 500,
    messages: [{
      role: 'user',
      content: \`Extract from this paper:
      - Title
      - Authors (list)
      - Abstract

      Text: \${text}

      Return as JSON: {"title": "", "authors": [], "abstract": ""}\`
    }]
  });
  return JSON.parse(response.content[0].text);
}

function chunkText(text: string, chunkSize: number, overlap: number) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push({
      text: text.slice(start, end),
      page: Math.floor(start / 3000) + 1  // Rough page estimate
    });
    start += chunkSize - overlap;
  }

  return chunks;
}

async function getEmbedding(text: string) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  });
  return response.data[0].embedding;
}
      </div>

      <h3>Step 4: RAG Query System</h3>
      <div class="code-block">
// lib/rag.ts
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

export async function queryPapers(
  question: string,
  paperIds: string[]
): Promise<string> {
  // Get question embedding
  const queryEmbedding = await getEmbedding(question);

  // Retrieve relevant chunks
  const { data: chunks } = await supabase.rpc('match_paper_chunks', {
    query_embedding: queryEmbedding,
    paper_ids: paperIds,
    match_count: 5
  });

  // Build context
  const context = chunks.map(c =>
    \`[Paper: \${c.paper_id}]\\n\${c.content}\`
  ).join('\\n\\n---\\n\\n');

  // Generate answer
  const response = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 2048,
    messages: [{
      role: 'user',
      content: \`You are a research assistant. Answer based on these paper excerpts:

\${context}

Question: \${question}

Instructions:
- Cite which paper(s) the information comes from
- If the answer isn't in the excerpts, say so
- Be precise and academic in tone\`
    }]
  });

  return response.content[0].text;
}

export async function comparePapers(paperIds: string[]): Promise<string> {
  // Get all papers' abstracts and key points
  const { data: papers } = await supabase
    .from('papers')
    .select('id, title, abstract')
    .in('id', paperIds);

  const comparison = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 3000,
    messages: [{
      role: 'user',
      content: \`Compare these research papers:

\${papers.map(p => \`## \${p.title}\\n\${p.abstract}\`).join('\\n\\n')}

Provide:
1. Common themes
2. Key differences in approach
3. Complementary findings
4. Contradictions (if any)
5. Research gaps identified\`
    }]
  });

  return comparison.content[0].text;
}
      </div>

      <h3>Step 5: API Routes</h3>
      <div class="code-block">
// app/api/papers/upload/route.ts
import { NextResponse } from 'next/server';
import { processPaper } from '@/lib/pdf-processor';
import { createClient } from '@/lib/supabase-server';

export async function POST(request: Request) {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const formData = await request.formData();
  const file = formData.get('file') as File;

  // Upload file to storage
  const { data: upload } = await supabase.storage
    .from('papers')
    .upload(\`\${user.id}/\${file.name}\`, file);

  // Create paper record
  const { data: paper } = await supabase
    .from('papers')
    .insert({
      user_id: user.id,
      title: file.name,
      file_url: upload.path
    })
    .select()
    .single();

  // Process in background
  const buffer = Buffer.from(await file.arrayBuffer());
  const metadata = await processPaper(buffer, paper.id);

  // Update with metadata
  await supabase
    .from('papers')
    .update(metadata)
    .eq('id', paper.id);

  return NextResponse.json({ paper });
}
      </div>

      <h3>Step 6: Chat Interface</h3>
      <div class="code-block">
// app/papers/[id]/chat/page.tsx
'use client';
import { useState } from 'react';

export default function PaperChat({ params }: { params: { id: string } }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    const response = await fetch('/api/papers/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: input,
        paperIds: [params.id]
      })
    });

    const data = await response.json();
    setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    setLoading(false);
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={\`p-4 rounded-lg \${
            msg.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'
          } max-w-2xl\`}>
            {msg.content}
          </div>
        ))}
        {loading && <div className="animate-pulse">Thinking...</div>}
      </div>
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about this paper..."
            className="flex-1 p-2 border rounded"
          />
          <button onClick={sendMessage} className="px-4 py-2 bg-blue-500 text-white rounded">
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
      </div>

      <h3>Deployment Checklist</h3>
      <ul>
        <li>☐ Set environment variables in Vercel</li>
        <li>☐ Configure Supabase Auth providers</li>
        <li>☐ Set up Supabase Storage bucket policies</li>
        <li>☐ Test PDF upload and processing</li>
        <li>☐ Verify RAG retrieval quality</li>
        <li>☐ Add error handling and loading states</li>
        <li>☐ Implement rate limiting</li>
        <li>☐ Add usage tracking and analytics</li>
      </ul>

      <h3>Extension Ideas</h3>
      <ul>
        <li>Add citation generation (BibTeX, APA)</li>
        <li>Implement paper recommendations</li>
        <li>Add collaborative features (shared libraries)</li>
        <li>Generate literature review drafts</li>
        <li>Extract and visualize key figures</li>
        <li>Build browser extension for one-click saving</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. You now have the skills to build complete AI applications</p>
        <p>2. RAG enables domain-specific AI without fine-tuning</p>
        <p>3. Proper chunking and embedding are crucial for retrieval quality</p>
        <p>4. Always consider UX: loading states, error handling, streaming</p>
        <p>5. Start simple, iterate, and add features based on user feedback</p>
      </div>
    `,
    questions: [
      {
        id: "20-1",
        type: "mcq",
        question: "What is the main advantage of using RAG over fine-tuning for a research assistant?",
        options: [
          "RAG is faster to train",
          "RAG allows real-time updates to the knowledge base without retraining",
          "RAG is more accurate",
          "RAG uses less memory"
        ],
        correctAnswer: "RAG allows real-time updates to the knowledge base without retraining",
        explanation: "RAG's main advantage is that you can add, update, or remove documents from the knowledge base instantly without any model training. This is perfect for a research assistant where papers are constantly being added."
      },
      {
        id: "20-2",
        type: "descriptive",
        question: "You've completed the AI course! Describe how you would build an AI agent that can help users with a specific task of your choice. Include the architecture, tools, and techniques you would use.",
        keywords: ["LLM", "tools", "RAG", "memory", "prompt", "API", "vector", "embedding", "user interface", "deployment"],
        explanation: "A complete answer should include: 1) Clear problem definition, 2) LLM selection for reasoning, 3) Tools the agent needs (APIs, search, etc.), 4) Memory system (vector DB for context), 5) RAG for domain knowledge, 6) Prompt engineering for reliability, 7) User interface design, 8) Deployment and monitoring plan."
      }
    ]
  }
];
