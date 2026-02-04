import { Lesson } from './lessons';

export const advancedLessons: Lesson[] = [
  {
    id: 18,
    title: "Transformer Architecture Deep Dive",
    description: "Master the complete transformer architecture - attention mechanisms, positional encoding, and implementation details",
    duration: "90 min",
    difficulty: "Advanced",
    content: `
      <h2>The Transformer Architecture in Detail</h2>
      <p>The Transformer, introduced in "Attention Is All You Need" (2017), revolutionized AI. Understanding it deeply is essential for any AI engineer.</p>

      <h3>Why Transformers Replaced RNNs</h3>
      <ul>
        <li><strong>Parallelization</strong> - RNNs process sequentially; Transformers process all positions simultaneously</li>
        <li><strong>Long-range dependencies</strong> - Attention connects any two positions directly</li>
        <li><strong>Scalability</strong> - Transformers scale better with more compute and data</li>
      </ul>

      <h3>Complete Architecture Overview</h3>
      <div class="code-block">
Input Tokens
    ↓
[Token Embedding + Positional Encoding]
    ↓
┌─────────────────────────────────┐
│     ENCODER (N layers)          │
│  ┌───────────────────────────┐  │
│  │ Multi-Head Self-Attention │  │
│  │      + Add & Norm         │  │
│  ├───────────────────────────┤  │
│  │ Feed-Forward Network      │  │
│  │      + Add & Norm         │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│     DECODER (N layers)          │
│  ┌───────────────────────────┐  │
│  │ Masked Self-Attention     │  │
│  │      + Add & Norm         │  │
│  ├───────────────────────────┤  │
│  │ Cross-Attention           │  │
│  │      + Add & Norm         │  │
│  ├───────────────────────────┤  │
│  │ Feed-Forward Network      │  │
│  │      + Add & Norm         │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
[Linear + Softmax]
    ↓
Output Probabilities
      </div>

      <h3>1. Input Embeddings</h3>
      <p>Converting tokens to dense vectors:</p>
      <div class="code-block">
# Token embedding
embedding_dim = 512
vocab_size = 50000
embedding_matrix = nn.Embedding(vocab_size, embedding_dim)

# For input tokens [5, 127, 89, 2004]
token_embeddings = embedding_matrix(tokens)  # Shape: (4, 512)
      </div>

      <h3>2. Positional Encoding</h3>
      <p>Since attention has no inherent sense of position, we add positional information:</p>

      <h4>Sinusoidal Positional Encoding (Original)</h4>
      <div class="code-block">
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index
- d_model = embedding dimension (512)
      </div>

      <div class="code-block">
import torch
import math

def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    return pe
      </div>

      <h4>Learned Positional Embeddings (Modern)</h4>
      <p>GPT and many modern models learn position embeddings:</p>
      <div class="code-block">
position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
positions = torch.arange(sequence_length)
pos_embeddings = position_embedding(positions)

# Final input = token_embeddings + pos_embeddings
      </div>

      <h3>3. Self-Attention Mechanism</h3>
      <p>The core innovation - allowing each position to attend to all positions:</p>

      <h4>Query, Key, Value Concept</h4>
      <ul>
        <li><strong>Query (Q)</strong> - "What am I looking for?"</li>
        <li><strong>Key (K)</strong> - "What do I contain that might be relevant?"</li>
        <li><strong>Value (V)</strong> - "What information do I provide?"</li>
      </ul>

      <h4>Attention Formula</h4>
      <div class="code-block">
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q, K, V are matrices of queries, keys, values
- d_k is the dimension of keys (for scaling)
- √d_k prevents dot products from getting too large
      </div>

      <h4>Step-by-Step Calculation</h4>
      <div class="code-block">
# Input: X (sequence_length × d_model)
# Weight matrices: W_Q, W_K, W_V (d_model × d_k)

# 1. Linear projections
Q = X @ W_Q  # (seq_len × d_k)
K = X @ W_K  # (seq_len × d_k)
V = X @ W_V  # (seq_len × d_v)

# 2. Compute attention scores
scores = Q @ K.T  # (seq_len × seq_len)

# 3. Scale
scores = scores / math.sqrt(d_k)

# 4. Apply softmax (row-wise)
attention_weights = softmax(scores, dim=-1)

# 5. Weighted sum of values
output = attention_weights @ V  # (seq_len × d_v)
      </div>

      <h4>Attention Visualization</h4>
      <p>For "The cat sat on the mat":</p>
      <div class="code-block">
              The   cat   sat   on   the   mat
The     [0.1   0.3   0.2   0.1  0.1   0.2]
cat     [0.2   0.4   0.1   0.1  0.1   0.1]
sat     [0.1   0.3   0.3   0.1  0.1   0.1]
on      [0.1   0.1   0.2   0.2  0.2   0.2]
the     [0.1   0.1   0.1   0.2  0.2   0.3]
mat     [0.1   0.2   0.2   0.1  0.2   0.2]

Each row shows how much that word attends to others
      </div>

      <h3>4. Multi-Head Attention</h3>
      <p>Running multiple attention operations in parallel:</p>
      <div class="code-block">
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

Where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
      </div>

      <h4>Why Multiple Heads?</h4>
      <ul>
        <li>Different heads can learn different relationship types</li>
        <li>Head 1 might learn syntax, Head 2 might learn coreference</li>
        <li>Typical: 8-96 heads depending on model size</li>
      </ul>

      <div class="code-block">
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)

        # Apply to values and reshape
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_O(out)
      </div>

      <h3>5. Feed-Forward Network</h3>
      <p>Applied to each position independently:</p>
      <div class="code-block">
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

# Typically d_ff = 4 × d_model
# d_model = 512 → d_ff = 2048
      </div>

      <div class="code-block">
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
      </div>

      <h3>6. Layer Normalization & Residual Connections</h3>
      <div class="code-block">
# Residual connection + Layer Norm
output = LayerNorm(x + Sublayer(x))

# Pre-norm (modern, more stable):
output = x + Sublayer(LayerNorm(x))
      </div>

      <h3>7. Masked Attention (Decoder)</h3>
      <p>Prevents attending to future positions during training:</p>
      <div class="code-block">
def create_causal_mask(size):
    # Lower triangular matrix
    mask = torch.tril(torch.ones(size, size))
    return mask

# For sequence length 4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
      </div>

      <h3>8. Complete Transformer Block</h3>
      <div class="code-block">
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
      </div>

      <h3>Key Hyperparameters</h3>
      <table>
        <tr><th>Parameter</th><th>GPT-2 Small</th><th>GPT-3</th><th>GPT-4 (est.)</th></tr>
        <tr><td>d_model</td><td>768</td><td>12288</td><td>~16384</td></tr>
        <tr><td>num_layers</td><td>12</td><td>96</td><td>~120</td></tr>
        <tr><td>num_heads</td><td>12</td><td>96</td><td>~128</td></tr>
        <tr><td>d_ff</td><td>3072</td><td>49152</td><td>~65536</td></tr>
        <tr><td>Parameters</td><td>117M</td><td>175B</td><td>~1.7T</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Self-attention allows every position to attend to every other position</p>
        <p>2. Multi-head attention learns different types of relationships in parallel</p>
        <p>3. Positional encoding adds sequence order information</p>
        <p>4. Residual connections and layer norm enable training deep networks</p>
        <p>5. The decoder uses causal masking to prevent looking at future tokens</p>
      </div>
    `,
    questions: [
      {
        id: "11-1",
        type: "mcq",
        question: "Why do we divide by √d_k in the attention formula?",
        options: [
          "To make computation faster",
          "To prevent dot products from becoming too large, which would cause softmax to have very small gradients",
          "To normalize the output to unit length",
          "To reduce memory usage"
        ],
        correctAnswer: "To prevent dot products from becoming too large, which would cause softmax to have very small gradients",
        explanation: "When d_k is large, dot products can become very large, pushing softmax into regions with tiny gradients. Scaling by √d_k keeps values in a reasonable range for stable training."
      },
      {
        id: "11-2",
        type: "mcq",
        question: "What is the purpose of multi-head attention?",
        options: [
          "To reduce computation time",
          "To allow the model to jointly attend to information from different representation subspaces",
          "To increase the sequence length",
          "To reduce the number of parameters"
        ],
        correctAnswer: "To allow the model to jointly attend to information from different representation subspaces",
        explanation: "Multi-head attention runs multiple attention operations in parallel, each potentially learning different types of relationships (syntax, semantics, coreference, etc.)."
      },
      {
        id: "11-3",
        type: "mcq",
        question: "Why does the decoder use masked self-attention?",
        options: [
          "To speed up training",
          "To prevent the model from attending to future tokens during training",
          "To reduce memory usage",
          "To improve accuracy"
        ],
        correctAnswer: "To prevent the model from attending to future tokens during training",
        explanation: "During training, the decoder sees the full target sequence but should only use past tokens to predict the next one. Masking ensures it can't 'cheat' by looking ahead."
      },
      {
        id: "11-4",
        type: "descriptive",
        question: "Explain the Query, Key, Value mechanism in attention. How does it allow the model to focus on relevant information?",
        keywords: ["query", "key", "value", "dot product", "similarity", "softmax", "weighted", "sum", "relevance", "attention scores"],
        explanation: "Q (query) represents what the current position is looking for. K (key) represents what each position offers. Their dot product gives similarity scores. Softmax normalizes these into weights. V (value) contains the actual information, and the weighted sum of values (by attention weights) gives the output."
      },
      {
        id: "11-5",
        type: "descriptive",
        question: "Write pseudocode for computing single-head self-attention given input X and weight matrices W_Q, W_K, W_V.",
        keywords: ["Q = X", "K = X", "V = X", "matmul", "softmax", "sqrt", "d_k", "scores", "weights", "output"],
        explanation: "Q = X @ W_Q; K = X @ W_K; V = X @ W_V; scores = (Q @ K.T) / sqrt(d_k); weights = softmax(scores); output = weights @ V"
      }
    ]
  },
  {
    id: 19,
    title: "Building AI Agents from Scratch",
    description: "Learn to build autonomous AI agents with tools, memory, and planning capabilities",
    duration: "120 min",
    difficulty: "Advanced",
    content: `
      <h2>What is an AI Agent?</h2>
      <p>An AI agent is a system that can perceive its environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain memory, and plan multi-step solutions.</p>

      <h3>Agent Architecture</h3>
      <div class="code-block">
┌─────────────────────────────────────────────────────────┐
│                      AI AGENT                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    LLM      │  │   Memory    │  │   Tools     │     │
│  │   (Brain)   │  │  (Context)  │  │  (Actions)  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         ↓                ↓                ↓             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Planning & Reasoning                │   │
│  │  • Decompose tasks into steps                   │   │
│  │  • Select appropriate tools                     │   │
│  │  • Handle errors and adapt                      │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Execution Loop                      │   │
│  │  Think → Act → Observe → Repeat                 │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
      </div>

      <h3>Core Components</h3>

      <h4>1. The LLM (Brain)</h4>
      <p>The reasoning engine that:</p>
      <ul>
        <li>Understands user requests</li>
        <li>Plans how to accomplish tasks</li>
        <li>Decides which tools to use</li>
        <li>Interprets tool outputs</li>
        <li>Generates final responses</li>
      </ul>

      <h4>2. Tools (Actions)</h4>
      <p>Functions the agent can call:</p>
      <div class="code-block">
# Tool definition
tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression"}
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "File path"}
        }
    }
]
      </div>

      <h4>3. Memory</h4>
      <p>Different types of memory:</p>
      <ul>
        <li><strong>Short-term (Working)</strong> - Current conversation context</li>
        <li><strong>Long-term (Episodic)</strong> - Past interactions, stored in vector DB</li>
        <li><strong>Semantic</strong> - Facts and knowledge</li>
        <li><strong>Procedural</strong> - How to perform tasks</li>
      </ul>

      <h3>The ReAct Pattern</h3>
      <p>Reasoning + Acting - the most common agent pattern:</p>
      <div class="code-block">
Loop:
  1. THOUGHT: Reason about what to do next
  2. ACTION: Choose and execute a tool
  3. OBSERVATION: See the result
  4. Repeat until task is complete
      </div>

      <h4>ReAct Prompt Template</h4>
      <div class="code-block">
You are an AI assistant that can use tools to help users.

Available tools:
{tool_descriptions}

Use this format:

Question: the user's question
Thought: reason about what to do
Action: tool_name
Action Input: {"param": "value"}
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the user's question
      </div>

      <h3>Building a Simple Agent</h3>

      <h4>Step 1: Define Tools</h4>
      <div class="code-block">
import json
import requests
from datetime import datetime

class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs):
        return self.func(**kwargs)

# Define tool functions
def web_search(query: str) -> str:
    # In production, use a real search API
    return f"Search results for: {query}"

def calculator(expression: str) -> str:
    try:
        result = eval(expression)  # Use safer evaluation in production
        return str(result)
    except:
        return "Error: Invalid expression"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create tool instances
tools = [
    Tool("web_search", "Search the web for information", web_search),
    Tool("calculator", "Calculate math expressions", calculator),
    Tool("get_time", "Get current date and time", get_current_time),
]
      </div>

      <h4>Step 2: Create the Agent Class</h4>
      <div class="code-block">
import anthropic

class Agent:
    def __init__(self, tools, model="claude-3-opus-20240229"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []
        self.max_iterations = 10

    def get_tool_descriptions(self):
        return "\\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

    def build_system_prompt(self):
        return f"""You are an AI assistant that can use tools.

Available tools:
{self.get_tool_descriptions()}

When you need to use a tool, respond with:
TOOL: tool_name
INPUT: {{"param": "value"}}

When you have the final answer, respond with:
ANSWER: your final answer

Always think step by step before acting."""

    def parse_response(self, response):
        text = response.content[0].text

        if "TOOL:" in text:
            lines = text.split("\\n")
            tool_name = None
            tool_input = None

            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("INPUT:"):
                    tool_input = json.loads(line.replace("INPUT:", "").strip())

            return {"type": "tool", "name": tool_name, "input": tool_input}

        elif "ANSWER:" in text:
            answer = text.split("ANSWER:")[-1].strip()
            return {"type": "answer", "content": answer}

        return {"type": "thinking", "content": text}

    def run(self, user_input):
        self.memory.append({"role": "user", "content": user_input})

        for i in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.build_system_prompt(),
                messages=self.memory
            )

            parsed = self.parse_response(response)

            if parsed["type"] == "answer":
                self.memory.append({
                    "role": "assistant",
                    "content": response.content[0].text
                })
                return parsed["content"]

            elif parsed["type"] == "tool":
                # Execute tool
                tool = self.tools.get(parsed["name"])
                if tool:
                    result = tool.run(**parsed["input"])
                    observation = f"Tool {parsed['name']} returned: {result}"
                else:
                    observation = f"Error: Tool {parsed['name']} not found"

                # Add to memory
                self.memory.append({
                    "role": "assistant",
                    "content": response.content[0].text
                })
                self.memory.append({
                    "role": "user",
                    "content": f"OBSERVATION: {observation}"
                })

            else:
                # Just thinking, continue
                self.memory.append({
                    "role": "assistant",
                    "content": response.content[0].text
                })

        return "Max iterations reached without final answer"
      </div>

      <h4>Step 3: Run the Agent</h4>
      <div class="code-block">
# Create and run agent
agent = Agent(tools)

# Example usage
result = agent.run("What is 25 * 47 + 123?")
print(result)

result = agent.run("What time is it and what's the weather in Tokyo?")
print(result)
      </div>

      <h3>Advanced Agent Patterns</h3>

      <h4>1. Planning Agent</h4>
      <p>Creates a plan before executing:</p>
      <div class="code-block">
def create_plan(self, task):
    prompt = f"""Create a step-by-step plan for: {task}

Format:
1. [Step description]
2. [Step description]
...

Only list the steps, don't execute them yet."""

    response = self.llm.generate(prompt)
    steps = self.parse_plan(response)
    return steps

def execute_plan(self, steps):
    results = []
    for step in steps:
        result = self.execute_step(step)
        results.append(result)

        # Check if we need to replan
        if self.should_replan(result):
            new_steps = self.replan(step, result)
            steps = new_steps + steps[steps.index(step)+1:]

    return results
      </div>

      <h4>2. Multi-Agent Systems</h4>
      <p>Multiple specialized agents working together:</p>
      <div class="code-block">
class MultiAgentSystem:
    def __init__(self):
        self.researcher = Agent(research_tools, persona="researcher")
        self.writer = Agent(writing_tools, persona="writer")
        self.critic = Agent([], persona="critic")
        self.coordinator = Agent([], persona="coordinator")

    def run(self, task):
        # Coordinator breaks down task
        subtasks = self.coordinator.decompose(task)

        # Research phase
        research = self.researcher.run(subtasks["research"])

        # Writing phase
        draft = self.writer.run(subtasks["write"], context=research)

        # Review phase
        feedback = self.critic.run(f"Review this: {draft}")

        # Revision
        final = self.writer.run(f"Revise based on: {feedback}")

        return final
      </div>

      <h4>3. Tool-Using Agent with Claude</h4>
      <div class="code-block">
# Using Claude's native tool use
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

# Handle tool use in response
for block in response.content:
    if block.type == "tool_use":
        tool_name = block.name
        tool_input = block.input
        # Execute tool and continue conversation
      </div>

      <h3>Memory Systems</h3>

      <h4>Vector Memory with Embeddings</h4>
      <div class="code-block">
from openai import OpenAI
import numpy as np

class VectorMemory:
    def __init__(self):
        self.client = OpenAI()
        self.memories = []
        self.embeddings = []

    def add(self, text):
        embedding = self.get_embedding(text)
        self.memories.append(text)
        self.embeddings.append(embedding)

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def search(self, query, top_k=5):
        query_embedding = self.get_embedding(query)

        # Calculate similarities
        similarities = [
            np.dot(query_embedding, emb) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]

        # Get top-k
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memories[i] for i in indices]
      </div>

      <h3>Error Handling & Reliability</h3>
      <div class="code-block">
class RobustAgent(Agent):
    def run_with_retry(self, user_input, max_retries=3):
        for attempt in range(max_retries):
            try:
                result = self.run(user_input)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.memory.append({
                    "role": "user",
                    "content": f"Error occurred: {str(e)}. Please try again."
                })

    def validate_tool_call(self, tool_name, tool_input):
        """Validate before executing potentially dangerous tools"""
        dangerous_tools = ["execute_code", "delete_file", "send_email"]

        if tool_name in dangerous_tools:
            # Require confirmation or additional checks
            return self.request_confirmation(tool_name, tool_input)
        return True
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Agents combine LLMs with tools, memory, and planning</p>
        <p>2. The ReAct pattern (Reason + Act) is fundamental</p>
        <p>3. Tools give agents capabilities beyond text generation</p>
        <p>4. Memory enables learning from past interactions</p>
        <p>5. Multi-agent systems can handle complex tasks through collaboration</p>
      </div>
    `,
    questions: [
      {
        id: "12-1",
        type: "mcq",
        question: "What is the ReAct pattern in AI agents?",
        options: [
          "A way to train neural networks faster",
          "A loop of Reasoning, Acting, and Observing results",
          "A method for reducing model size",
          "A technique for data augmentation"
        ],
        correctAnswer: "A loop of Reasoning, Acting, and Observing results",
        explanation: "ReAct (Reasoning + Acting) is a pattern where the agent thinks about what to do, takes an action (usually using a tool), observes the result, and repeats until the task is complete."
      },
      {
        id: "12-2",
        type: "mcq",
        question: "What are the core components of an AI agent?",
        options: [
          "Only an LLM",
          "LLM, Tools, and Memory",
          "Database and API",
          "Frontend and Backend"
        ],
        correctAnswer: "LLM, Tools, and Memory",
        explanation: "A complete AI agent typically has: an LLM as the reasoning engine, tools for taking actions in the world, and memory for maintaining context and learning from past interactions."
      },
      {
        id: "12-3",
        type: "descriptive",
        question: "Explain how vector memory works in an AI agent and why it's useful.",
        keywords: ["embedding", "vector", "similarity", "search", "semantic", "store", "retrieve", "context", "long-term", "relevant"],
        explanation: "Vector memory converts text to embeddings (dense vectors), stores them, and retrieves relevant memories by computing similarity between query and stored embeddings. This enables semantic search - finding contextually relevant past information even if exact words differ."
      },
      {
        id: "12-4",
        type: "descriptive",
        question: "Describe how you would implement a tool for an AI agent. Include the tool definition and execution.",
        keywords: ["name", "description", "parameters", "function", "execute", "input", "output", "schema", "run", "return"],
        explanation: "A tool needs: name, description (for LLM to understand when to use it), parameter schema (types, descriptions), and an execution function. The agent calls the tool with parameters, the function executes and returns results, which the agent uses to continue reasoning."
      }
    ]
  }
];
