import { Lesson } from './lessons';

// Premium lessons to elevate course quality to 8.5+/10
export const premiumLessons: Lesson[] = [
  {
    id: 20,
    title: "Convolutional Neural Networks (CNNs)",
    description: "Master image processing with CNNs - the backbone of computer vision AI",
    duration: "75 min",
    difficulty: "Intermediate",
    content: `
      <h2>Why CNNs Revolutionized Computer Vision</h2>
      <p>Before CNNs, computers struggled to "see." Convolutional Neural Networks changed everything by mimicking how the human visual cortex processes images - detecting edges, then shapes, then objects.</p>

      <div class="highlight">
        <p><strong>Key Insight:</strong> CNNs exploit the spatial structure of images. A pixel's meaning depends on its neighbors, not its absolute position.</p>
      </div>

      <h3>The Problem with Regular Neural Networks for Images</h3>
      <div class="code-block">
        <pre>
# A small 224×224 RGB image has:
224 × 224 × 3 = 150,528 input values

# Fully connected layer with 1000 neurons:
150,528 × 1000 = 150 MILLION parameters!

# Problems:
# 1. Way too many parameters → overfitting
# 2. Ignores spatial relationships
# 3. Not translation invariant (cat in corner ≠ cat in center)
        </pre>
      </div>

      <h3>The Convolution Operation</h3>
      <p>A <strong>convolution</strong> slides a small filter (kernel) across the image, computing dot products at each position.</p>

      <div class="code-block">
        <pre>
# Example: 3×3 edge detection filter
filter = [
  [-1, -1, -1],
  [-1,  8, -1],
  [-1, -1, -1]
]

# Slide across image:
# At each position, multiply filter with image patch
# Sum the results → one output value

# Input: 6×6 image
# Filter: 3×3
# Output: 4×4 feature map (6-3+1 = 4)
        </pre>
      </div>

      <h4>What Filters Learn to Detect</h4>
      <ul>
        <li><strong>Early layers:</strong> Edges, colors, gradients</li>
        <li><strong>Middle layers:</strong> Textures, patterns, parts (eyes, wheels)</li>
        <li><strong>Deep layers:</strong> Objects, faces, scenes</li>
      </ul>

      <h3>Key CNN Components</h3>

      <h4>1. Convolutional Layer</h4>
      <div class="code-block">
        <pre>
import torch.nn as nn

# Convolutional layer
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 different filters
    kernel_size=3,      # 3×3 filters
    stride=1,           # Move 1 pixel at a time
    padding=1           # Pad edges to maintain size
)

# Parameters: 3 × 64 × 3 × 3 = 1,728 (not millions!)
        </pre>
      </div>

      <h4>2. Pooling Layer (Downsampling)</h4>
      <div class="code-block">
        <pre>
# Max Pooling: Take maximum value in each region
# Reduces spatial dimensions, keeps important features

# Example: 2×2 max pooling
[1, 3, 2, 4]      [3, 4]
[5, 6, 7, 8]  →   [6, 8]
[1, 2, 3, 4]
[5, 6, 7, 8]

# Input: 4×4 → Output: 2×2
# Reduces computation, adds translation invariance

pool = nn.MaxPool2d(kernel_size=2, stride=2)
        </pre>
      </div>

      <h4>3. Batch Normalization</h4>
      <div class="code-block">
        <pre>
# Normalizes activations → faster, more stable training
# Applied after convolution, before activation

bn = nn.BatchNorm2d(num_features=64)

# Forward pass:
x = conv(input)
x = bn(x)
x = relu(x)
        </pre>
      </div>

      <h4>4. Fully Connected Layers (at the end)</h4>
      <div class="code-block">
        <pre>
# After convolutions extract features:
# Flatten → FC layers → Output predictions

x = x.view(batch_size, -1)  # Flatten
x = fc1(x)                   # Dense layer
x = fc2(x)                   # Output layer (num_classes)
        </pre>
      </div>

      <h3>Classic CNN Architectures</h3>

      <h4>LeNet-5 (1998) - The Pioneer</h4>
      <div class="code-block">
        <pre>
# Simple architecture for digit recognition
Input (32×32) → Conv → Pool → Conv → Pool → FC → FC → Output
        </pre>
      </div>

      <h4>AlexNet (2012) - The Breakthrough</h4>
      <div class="code-block">
        <pre>
# Won ImageNet, sparked deep learning revolution
# Key innovations:
# - ReLU activation (faster training)
# - Dropout (reduces overfitting)
# - GPU training
# - Data augmentation

Conv(96) → Pool → Conv(256) → Pool → Conv(384) → Conv(384) → Conv(256) → Pool → FC → FC → Output
        </pre>
      </div>

      <h4>VGG (2014) - Deeper is Better</h4>
      <div class="code-block">
        <pre>
# Very deep (16-19 layers) with small 3×3 filters
# Key insight: Stack of small filters = large receptive field
# 2 × (3×3) has same receptive field as 1 × (5×5)
# But fewer parameters and more non-linearity

VGG-16: 13 conv layers + 3 FC layers
        </pre>
      </div>

      <h4>ResNet (2015) - Skip Connections</h4>
      <div class="code-block">
        <pre>
# Problem: Very deep networks hard to train (vanishing gradients)
# Solution: Skip connections (residual connections)

# Instead of learning H(x), learn F(x) = H(x) - x
# Output = F(x) + x (skip connection)

class ResidualBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip connection!
        out = self.relu(out)
        return out

# Enables training of 100+ layer networks!
# ResNet-50, ResNet-101, ResNet-152
        </pre>
      </div>

      <h3>Building a CNN from Scratch</h3>
      <div class="code-block">
        <pre>
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Input: 32×32×3 image
# After block 1: 16×16×32
# After block 2: 8×8×64
# After block 3: 4×4×128
# Flatten: 2048 → 256 → num_classes
        </pre>
      </div>

      <h3>Data Augmentation</h3>
      <p>CNNs need lots of data. Augmentation creates variations artificially:</p>
      <div class="code-block">
        <pre>
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 50% chance flip
    transforms.RandomRotation(10),          # ±10 degrees
    transforms.RandomResizedCrop(224),      # Random crop & resize
    transforms.ColorJitter(brightness=0.2), # Vary brightness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# One image becomes many variations → reduces overfitting
        </pre>
      </div>

      <h3>When to Use CNNs</h3>
      <table>
        <tr><th>Use CNNs</th><th>Don't Use CNNs</th></tr>
        <tr><td>Image classification</td><td>Tabular data</td></tr>
        <tr><td>Object detection</td><td>Time series (use RNNs)</td></tr>
        <tr><td>Semantic segmentation</td><td>Text (use Transformers)</td></tr>
        <tr><td>Medical imaging</td><td>Small datasets (&lt;1000 images)</td></tr>
        <tr><td>Video analysis</td><td>Non-spatial data</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Convolutions</strong> detect local patterns with shared weights</p>
        <p>2. <strong>Pooling</strong> reduces dimensions and adds translation invariance</p>
        <p>3. <strong>Deeper layers</strong> detect more complex features</p>
        <p>4. <strong>Skip connections</strong> enable training very deep networks</p>
        <p>5. <strong>Data augmentation</strong> is essential to prevent overfitting</p>
      </div>
    `,
    questions: [
      {
        id: "20-1",
        type: "mcq",
        question: "Why do CNNs use convolutions instead of fully connected layers for images?",
        options: [
          "Convolutions are faster to compute",
          "Convolutions have fewer parameters and exploit spatial structure",
          "Fully connected layers can't process images",
          "Convolutions produce better colors"
        ],
        correctAnswer: "Convolutions have fewer parameters and exploit spatial structure",
        explanation: "CNNs use convolutions because: (1) They have far fewer parameters by sharing weights across positions, (2) They exploit spatial locality - neighboring pixels are related, (3) They're translation invariant - can detect a cat anywhere in the image."
      },
      {
        id: "20-2",
        type: "mcq",
        question: "What do early layers in a CNN typically learn to detect?",
        options: ["Complete objects", "Faces and animals", "Edges and simple patterns", "Text and numbers"],
        correctAnswer: "Edges and simple patterns",
        explanation: "CNNs learn hierarchically: early layers detect simple features (edges, colors, gradients), middle layers detect parts and textures, and deep layers detect complete objects and scenes."
      },
      {
        id: "20-3",
        type: "descriptive",
        question: "Explain what a residual (skip) connection is and why it helps train very deep networks.",
        keywords: ["skip", "identity", "gradient", "vanishing", "residual", "add", "shortcut", "deep", "layers", "learn"],
        explanation: "A skip connection adds the input directly to the output of a block: output = F(x) + x. This helps because: (1) Gradients can flow directly through the skip path, preventing vanishing gradients. (2) The network only needs to learn the 'residual' F(x) = H(x) - x, which is easier. (3) Enables training of 100+ layer networks like ResNet."
      },
      {
        id: "20-4",
        type: "mcq",
        question: "What is the purpose of max pooling in a CNN?",
        options: [
          "To increase image resolution",
          "To reduce spatial dimensions and provide translation invariance",
          "To add more filters",
          "To normalize the data"
        ],
        correctAnswer: "To reduce spatial dimensions and provide translation invariance",
        explanation: "Max pooling reduces spatial dimensions (e.g., 4×4 → 2×2) by taking the maximum value in each region. This reduces computation, provides some translation invariance (small shifts don't change the max), and helps prevent overfitting."
      }
    ]
  },
  {
    id: 21,
    title: "Recurrent Neural Networks & LSTMs",
    description: "Process sequential data with memory - essential for time series and language",
    duration: "70 min",
    difficulty: "Intermediate",
    content: `
      <h2>The Need for Sequence Models</h2>
      <p>Regular neural networks can't handle sequences well because:</p>
      <ul>
        <li>Input/output sizes are fixed</li>
        <li>No memory of previous inputs</li>
        <li>Each input processed independently</li>
      </ul>

      <p><strong>Sequential data is everywhere:</strong> text, speech, time series, video, DNA, music...</p>

      <div class="highlight">
        <p><strong>Key Insight:</strong> In sequences, context matters. "The bank by the river" vs "I went to the bank" - the meaning of "bank" depends on previous words.</p>
      </div>

      <h3>Recurrent Neural Networks (RNNs)</h3>
      <p>RNNs have a <strong>hidden state</strong> that acts as memory, updated at each time step:</p>

      <div class="code-block">
        <pre>
# RNN at each time step t:
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
y_t = W_hy × h_t

# Where:
# h_t = hidden state at time t (the "memory")
# x_t = input at time t
# y_t = output at time t
# W_hh, W_xh, W_hy = weight matrices (shared across time!)

# The same weights are used at every time step
# This is "weight sharing" across time
        </pre>
      </div>

      <h4>Unrolling an RNN</h4>
      <div class="code-block">
        <pre>
# Processing "hello":
# h_0 (initial) → [h_0, 'h'] → h_1
#                 [h_1, 'e'] → h_2
#                 [h_2, 'l'] → h_3
#                 [h_3, 'l'] → h_4
#                 [h_4, 'o'] → h_5 → output

# Each h contains information about all previous characters
# h_5 "remembers" the whole word
        </pre>
      </div>

      <h4>Simple RNN in PyTorch</h4>
      <div class="code-block">
        <pre>
import torch.nn as nn

# Simple RNN layer
rnn = nn.RNN(
    input_size=10,     # Size of each input element
    hidden_size=64,    # Size of hidden state
    num_layers=2,      # Stack 2 RNN layers
    batch_first=True   # Input shape: (batch, seq_len, features)
)

# Forward pass
# input shape: (batch_size, sequence_length, input_size)
output, hidden = rnn(input, h_0)

# output: all hidden states (batch, seq_len, hidden_size)
# hidden: final hidden state (num_layers, batch, hidden_size)
        </pre>
      </div>

      <h3>The Vanishing Gradient Problem</h3>
      <p>RNNs struggle with long sequences because gradients vanish during backpropagation:</p>

      <div class="code-block">
        <pre>
# Backpropagation through time (BPTT):
# Gradient at t=0 depends on product of many derivatives
# If each derivative < 1, product → 0 (vanishing)
# If each derivative > 1, product → ∞ (exploding)

# Example with 100 time steps:
# gradient ∝ 0.9^100 ≈ 0.0000027 (vanished!)
# gradient ∝ 1.1^100 ≈ 13,781 (exploded!)

# Result: RNN can't learn long-range dependencies
# "The cat, which was sitting on the mat, was ___"
# RNN forgets "cat" by the time it reaches the blank
        </pre>
      </div>

      <h3>LSTM: Long Short-Term Memory</h3>
      <p>LSTMs solve vanishing gradients with <strong>gating mechanisms</strong>:</p>

      <div class="code-block">
        <pre>
# LSTM has TWO states:
# - h_t: hidden state (short-term memory)
# - c_t: cell state (long-term memory, the "conveyor belt")

# THREE gates control information flow:

# 1. FORGET GATE: What to remove from cell state
f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)
# Output: 0 = forget completely, 1 = keep everything

# 2. INPUT GATE: What new info to add
i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
candidate = tanh(W_c × [h_{t-1}, x_t] + b_c)

# 3. OUTPUT GATE: What to output from cell state
o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)

# UPDATE CELL STATE:
c_t = f_t * c_{t-1} + i_t * candidate
# Old memory (possibly forgotten) + new candidates

# OUTPUT:
h_t = o_t * tanh(c_t)
        </pre>
      </div>

      <h4>Why LSTMs Work</h4>
      <div class="code-block">
        <pre>
# The cell state c_t is a "highway" for gradients
# If forget gate = 1, gradient flows unchanged:
# ∂c_t/∂c_{t-1} = f_t ≈ 1

# No vanishing gradient on the cell state path!
# Information can persist for hundreds of time steps
        </pre>
      </div>

      <h4>LSTM in PyTorch</h4>
      <div class="code-block">
        <pre>
import torch.nn as nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    batch_first=True,
    dropout=0.2,       # Dropout between layers
    bidirectional=True # Process forward and backward
)

# Forward pass
output, (h_n, c_n) = lstm(input, (h_0, c_0))

# For bidirectional:
# hidden_size becomes 128 (64 forward + 64 backward)
        </pre>
      </div>

      <h3>GRU: Gated Recurrent Unit</h3>
      <p>Simplified LSTM with only 2 gates (often works just as well):</p>

      <div class="code-block">
        <pre>
# GRU has:
# - Reset gate r_t: How much past to forget
# - Update gate z_t: How much to update

r_t = sigmoid(W_r × [h_{t-1}, x_t])
z_t = sigmoid(W_z × [h_{t-1}, x_t])

candidate = tanh(W × [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * candidate

# Fewer parameters than LSTM
# Often similar performance
# Use GRU when computational efficiency matters

gru = nn.GRU(input_size=10, hidden_size=64, batch_first=True)
        </pre>
      </div>

      <h3>Bidirectional RNNs</h3>
      <div class="code-block">
        <pre>
# Process sequence in both directions
# Forward:  "The cat sat" → h_forward
# Backward: "tas tac ehT" → h_backward

# Concatenate: h = [h_forward; h_backward]

# Why? Some tasks need future context:
# "The ___ barked loudly" (need "loudly" to know it's "dog")

lstm = nn.LSTM(hidden_size=64, bidirectional=True)
# Output hidden size: 64 × 2 = 128
        </pre>
      </div>

      <h3>Sequence-to-Sequence (Seq2Seq)</h3>
      <div class="code-block">
        <pre>
# For tasks like translation: variable input → variable output

# ENCODER: Process input sequence
encoder_outputs, (h_n, c_n) = encoder_lstm(source_sentence)

# DECODER: Generate output sequence
decoder_input = [START_TOKEN]
for step in range(max_length):
    output, (h_n, c_n) = decoder_lstm(decoder_input, (h_n, c_n))
    predicted_word = output.argmax()
    if predicted_word == END_TOKEN:
        break
    decoder_input = predicted_word

# Note: This is pre-Transformer architecture
# Transformers now dominate for translation
        </pre>
      </div>

      <h3>Practical Example: Sentiment Analysis</h3>
      <div class="code-block">
        <pre>
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, x):
        # x: (batch, seq_len) - token indices
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)  # h_n: (1, batch, hidden)
        h_n = h_n.squeeze(0)  # (batch, hidden)
        out = self.fc(h_n)  # (batch, 1)
        return torch.sigmoid(out)

# Use final hidden state for classification
# It contains information about the whole sequence
        </pre>
      </div>

      <h3>When to Use RNNs/LSTMs vs Transformers</h3>
      <table>
        <tr><th>Use RNNs/LSTMs</th><th>Use Transformers</th></tr>
        <tr><td>Time series forecasting</td><td>NLP tasks (translation, QA)</td></tr>
        <tr><td>Real-time streaming data</td><td>Large text datasets</td></tr>
        <tr><td>Limited compute resources</td><td>When parallelization needed</td></tr>
        <tr><td>Very long sequences (with attention)</td><td>Medium sequences (&lt;512 tokens)</td></tr>
        <tr><td>Audio/signal processing</td><td>Pre-training on massive corpora</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>RNNs</strong> have memory but suffer from vanishing gradients</p>
        <p>2. <strong>LSTMs</strong> use gates to control information flow and enable long-term memory</p>
        <p>3. <strong>GRUs</strong> are simplified LSTMs with similar performance</p>
        <p>4. <strong>Bidirectional</strong> RNNs capture both past and future context</p>
        <p>5. <strong>Transformers</strong> have largely replaced RNNs for NLP, but RNNs remain useful for time series</p>
      </div>
    `,
    questions: [
      {
        id: "21-1",
        type: "mcq",
        question: "What problem do LSTMs solve that regular RNNs have?",
        options: [
          "LSTMs are faster to train",
          "LSTMs solve the vanishing gradient problem",
          "LSTMs use less memory",
          "LSTMs work with images"
        ],
        correctAnswer: "LSTMs solve the vanishing gradient problem",
        explanation: "Regular RNNs suffer from vanishing gradients - when backpropagating through many time steps, gradients shrink to near zero, making it impossible to learn long-range dependencies. LSTMs solve this with the cell state 'highway' that allows gradients to flow unchanged."
      },
      {
        id: "21-2",
        type: "mcq",
        question: "What is the purpose of the forget gate in an LSTM?",
        options: [
          "To output the final prediction",
          "To decide what information to remove from the cell state",
          "To add new information to the cell state",
          "To initialize the hidden state"
        ],
        correctAnswer: "To decide what information to remove from the cell state",
        explanation: "The forget gate outputs values between 0 and 1 for each element in the cell state. A value of 0 means 'forget this completely' and 1 means 'keep this entirely.' This allows the LSTM to selectively discard irrelevant information."
      },
      {
        id: "21-3",
        type: "descriptive",
        question: "Explain the difference between the hidden state and cell state in an LSTM.",
        keywords: ["hidden", "cell", "short-term", "long-term", "memory", "output", "gate", "conveyor", "flow"],
        explanation: "The hidden state (h_t) is the short-term memory and output at each step - it's passed to the next layer or used for predictions. The cell state (c_t) is the long-term memory - a 'conveyor belt' that carries information across many time steps with minimal modification. The cell state enables LSTMs to remember information for hundreds of steps, while the hidden state is more like working memory."
      },
      {
        id: "21-4",
        type: "mcq",
        question: "When would you use a bidirectional LSTM?",
        options: [
          "When you need faster training",
          "When future context helps understand the current position",
          "When processing images",
          "When memory is limited"
        ],
        correctAnswer: "When future context helps understand the current position",
        explanation: "Bidirectional LSTMs process the sequence both forward and backward, then concatenate the hidden states. This is useful when understanding a position requires future context, like in named entity recognition ('The ___ barked' - knowing 'barked' helps identify the entity as a dog)."
      }
    ]
  },
  {
    id: 22,
    title: "Transfer Learning & Fine-Tuning",
    description: "Leverage pre-trained models to achieve state-of-the-art results with less data",
    duration: "60 min",
    difficulty: "Intermediate",
    content: `
      <h2>The Power of Transfer Learning</h2>
      <p>Training deep learning models from scratch requires:</p>
      <ul>
        <li>Millions of labeled examples</li>
        <li>Days/weeks of GPU training</li>
        <li>Expert knowledge to avoid pitfalls</li>
      </ul>

      <p><strong>Transfer learning</strong> lets you leverage models trained by others on massive datasets, adapting them to your specific task with minimal data and compute.</p>

      <div class="highlight">
        <p><strong>Key Insight:</strong> Features learned on one task often transfer to related tasks. A model trained on ImageNet learns general visual features (edges, textures, shapes) useful for almost any image task.</p>
      </div>

      <h3>How Transfer Learning Works</h3>
      <div class="code-block">
        <pre>
# Traditional ML:
Task A data → Train Model A → Use for Task A only

# Transfer Learning:
Task A (large dataset) → Train Model A →
  → Take learned features →
  → Fine-tune on Task B (small dataset) →
  → Great results on Task B!

# Example:
ImageNet (14M images) → Pre-trained ResNet →
  → Fine-tune on medical images (10K images) →
  → Accurate disease detection!
        </pre>
      </div>

      <h3>Transfer Learning Strategies</h3>

      <h4>Strategy 1: Feature Extraction (Freeze Base)</h4>
      <div class="code-block">
        <pre>
# Use pre-trained model as fixed feature extractor
# Only train the new classification head

import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# FREEZE all base layers
for param in model.parameters():
    param.requires_grad = False

# REPLACE classification head
num_classes = 10  # Your task
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only fc layer will be trained
# Fast training, works with very small datasets (100s of images)
        </pre>
      </div>

      <h4>Strategy 2: Fine-Tuning (Train Everything)</h4>
      <div class="code-block">
        <pre>
# Train the whole model with a small learning rate
# Better results but needs more data

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Use SMALL learning rate to not destroy pre-trained weights
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Alternative: Different learning rates per layer
# Higher for new layers, lower for pre-trained
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},      # New head
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Last block
    {'params': model.layer3.parameters(), 'lr': 1e-5},  # Earlier blocks
])
        </pre>
      </div>

      <h4>Strategy 3: Gradual Unfreezing</h4>
      <div class="code-block">
        <pre>
# Start frozen, gradually unfreeze layers

# Epoch 1-3: Train only head
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
train(epochs=3)

# Epoch 4-6: Unfreeze last block
for param in model.layer4.parameters():
    param.requires_grad = True
train(epochs=3)

# Epoch 7+: Unfreeze everything
for param in model.parameters():
    param.requires_grad = True
train(epochs=10)
        </pre>
      </div>

      <h3>Transfer Learning for Computer Vision</h3>
      <div class="code-block">
        <pre>
from torchvision import models, transforms

# Popular pre-trained models:
# - ResNet (good balance of accuracy/speed)
# - EfficientNet (best accuracy/params ratio)
# - VGG (simple, good for feature extraction)
# - MobileNet (fast, for mobile/edge)

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = models.efficientnet_b0(pretrained=True)

# Modify for your task
model.classifier[1] = nn.Linear(1280, num_classes)
        </pre>
      </div>

      <h3>Transfer Learning for NLP</h3>
      <div class="code-block">
        <pre>
from transformers import AutoModel, AutoTokenizer

# Popular pre-trained models:
# - BERT: Bidirectional understanding
# - RoBERTa: Optimized BERT
# - DistilBERT: Smaller, faster BERT
# - GPT-2/3: Generative models

# Load pre-trained BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

# For classification: Add head on [CLS] token
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)
        </pre>
      </div>

      <h3>Fine-Tuning Best Practices</h3>

      <h4>1. Learning Rate</h4>
      <div class="code-block">
        <pre>
# TOO HIGH: Destroys pre-trained weights ("catastrophic forgetting")
# TOO LOW: Takes forever, might not converge

# Good starting points:
# Feature extraction: 1e-3 to 1e-2 (new layers only)
# Fine-tuning: 1e-5 to 1e-4 (all layers)
# LLMs: 2e-5 (recommended by BERT paper)

# Use learning rate finder or start low
        </pre>
      </div>

      <h4>2. Batch Size</h4>
      <div class="code-block">
        <pre>
# Larger batches → more stable gradients
# But limited by GPU memory

# For fine-tuning, often use smaller batches:
# BERT: 16-32
# Vision: 32-64

# Gradient accumulation for effective larger batches:
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        </pre>
      </div>

      <h4>3. Data Augmentation</h4>
      <div class="code-block">
        <pre>
# More important with small datasets
# But don't go overboard - stay realistic

# Vision:
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# NLP:
# - Back-translation (translate to another language and back)
# - Synonym replacement
# - Random insertion/deletion
        </pre>
      </div>

      <h4>4. Early Stopping</h4>
      <div class="code-block">
        <pre>
# Fine-tuning can overfit quickly on small datasets
# Monitor validation loss, stop when it increases

best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
        </pre>
      </div>

      <h3>When Transfer Learning Helps Most</h3>
      <table>
        <tr><th>Scenario</th><th>Strategy</th><th>Expected Benefit</th></tr>
        <tr><td>Very small dataset (&lt;1K)</td><td>Feature extraction only</td><td>High</td></tr>
        <tr><td>Small dataset (1K-10K)</td><td>Fine-tune top layers</td><td>High</td></tr>
        <tr><td>Medium dataset (10K-100K)</td><td>Fine-tune all layers</td><td>Medium</td></tr>
        <tr><td>Large dataset (&gt;100K)</td><td>Train from scratch or fine-tune</td><td>Low-Medium</td></tr>
        <tr><td>Very different domain</td><td>Fine-tune carefully</td><td>Variable</td></tr>
      </table>

      <h3>Hugging Face Example</h3>
      <div class="code-block">
        <pre>
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load pre-trained model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Train with Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Transfer learning</strong> leverages pre-trained features for new tasks</p>
        <p>2. <strong>Feature extraction</strong> freezes base, trains only head - fast, works with little data</p>
        <p>3. <strong>Fine-tuning</strong> trains everything with small LR - better results, needs more data</p>
        <p>4. <strong>Learning rate</strong> is critical - too high destroys pre-trained weights</p>
        <p>5. <strong>Hugging Face</strong> makes transfer learning for NLP trivially easy</p>
      </div>
    `,
    questions: [
      {
        id: "22-1",
        type: "mcq",
        question: "When fine-tuning a pre-trained model, why should you use a small learning rate?",
        options: [
          "To speed up training",
          "To prevent destroying the useful pre-trained weights",
          "To use less memory",
          "To increase model size"
        ],
        correctAnswer: "To prevent destroying the useful pre-trained weights",
        explanation: "A large learning rate can cause 'catastrophic forgetting' - the model loses the useful features it learned during pre-training. A small learning rate (1e-5 to 1e-4) makes gentle updates that preserve pre-trained knowledge while adapting to the new task."
      },
      {
        id: "22-2",
        type: "mcq",
        question: "What is the difference between feature extraction and fine-tuning?",
        options: [
          "Feature extraction is faster because it only trains the new classification head",
          "Fine-tuning uses more data",
          "Feature extraction works better on large datasets",
          "They are the same thing"
        ],
        correctAnswer: "Feature extraction is faster because it only trains the new classification head",
        explanation: "Feature extraction freezes all pre-trained layers and only trains the new head - it's faster and works with very small datasets. Fine-tuning trains all layers (with a small LR) - it's slower but often achieves better results when you have more data."
      },
      {
        id: "22-3",
        type: "descriptive",
        question: "Explain when and why you would use gradual unfreezing in transfer learning.",
        keywords: ["layers", "unfreeze", "gradually", "epochs", "head", "catastrophic", "forgetting", "stable", "early", "deep"],
        explanation: "Gradual unfreezing: (1) Start with all pre-trained layers frozen, train only the new head. (2) After a few epochs, unfreeze the last block and continue training. (3) Progressively unfreeze earlier layers. This helps because: early layers learn general features (edges, textures) that transfer well, while deeper layers learn task-specific features that may need more adaptation. Gradual unfreezing prevents catastrophic forgetting and leads to more stable training."
      },
      {
        id: "22-4",
        type: "mcq",
        question: "Which transfer learning strategy is best for a very small dataset (a few hundred images)?",
        options: [
          "Train from scratch",
          "Fine-tune all layers",
          "Feature extraction (freeze base, train head only)",
          "Train only the first few layers"
        ],
        correctAnswer: "Feature extraction (freeze base, train head only)",
        explanation: "With very small datasets, fine-tuning risks overfitting because there's not enough data to update millions of parameters. Feature extraction works well because: (1) The pre-trained features are already useful, (2) Only a few parameters (the head) are trained, (3) Less prone to overfitting."
      }
    ]
  },
  {
    id: 23,
    title: "Generative AI: VAEs, GANs & Diffusion Models",
    description: "Understand how AI creates new images, art, and content",
    duration: "80 min",
    difficulty: "Advanced",
    content: `
      <h2>From Classification to Generation</h2>
      <p>Most of this course has focused on <strong>discriminative models</strong> - predicting labels from data. <strong>Generative models</strong> do the opposite - they learn the data distribution and can create new samples.</p>

      <div class="highlight">
        <p><strong>Discriminative:</strong> P(label | data) - "Is this a cat or dog?"</p>
        <p><strong>Generative:</strong> P(data) - "Create a new cat image"</p>
      </div>

      <h3>Why Generative Models Matter</h3>
      <ul>
        <li><strong>Art & Content Creation:</strong> DALL-E, Midjourney, Stable Diffusion</li>
        <li><strong>Data Augmentation:</strong> Generate training data for rare cases</li>
        <li><strong>Drug Discovery:</strong> Generate candidate molecules</li>
        <li><strong>Simulation:</strong> Generate realistic scenarios for training</li>
        <li><strong>Compression:</strong> Learn compact representations</li>
      </ul>

      <h3>Variational Autoencoders (VAEs)</h3>
      <p>VAEs learn to compress data into a latent space and reconstruct it.</p>

      <h4>The Autoencoder Foundation</h4>
      <div class="code-block">
        <pre>
# Regular Autoencoder:
# Input → Encoder → Latent Code (z) → Decoder → Reconstruction

# Goal: Reconstruction loss should be low
# loss = ||input - reconstruction||²

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Problem: Latent space is not continuous
# Random sampling produces garbage
        </pre>
      </div>

      <h4>VAE: Making Latent Space Continuous</h4>
      <div class="code-block">
        <pre>
# VAE key insight: Encode to a DISTRIBUTION, not a point
# z ~ N(μ, σ²) instead of z = encoder(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder outputs μ and log(σ²)
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, 64)
        self.fc_logvar = nn.Linear(256, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick for backprop
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# VAE Loss = Reconstruction + KL Divergence
# KL forces latent space to be close to N(0, 1)
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
        </pre>
      </div>

      <h4>Generating with VAEs</h4>
      <div class="code-block">
        <pre>
# Sample from standard normal
z = torch.randn(1, 64)

# Decode to generate new image
generated_image = vae.decoder(z)

# Interpolate between two images
z1 = vae.encode(image1)[0]  # Get mean
z2 = vae.encode(image2)[0]
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = alpha * z1 + (1 - alpha) * z2
    interpolated = vae.decoder(z_interp)
        </pre>
      </div>

      <h3>Generative Adversarial Networks (GANs)</h3>
      <p>GANs use two networks in competition to generate realistic samples.</p>

      <div class="code-block">
        <pre>
# Two players:
# Generator (G): Creates fake samples from random noise
# Discriminator (D): Tries to distinguish real from fake

# Training is a minimax game:
# G tries to MAXIMIZE D's error
# D tries to MINIMIZE classification error

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real
        )

    def forward(self, x):
        return self.model(x)
        </pre>
      </div>

      <h4>GAN Training Loop</h4>
      <div class="code-block">
        <pre>
for epoch in range(epochs):
    for real_images in dataloader:

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        d_loss_real = criterion(discriminator(real_images), real_labels)
        d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)

        # Generator wants D to think fakes are real
        g_loss = criterion(discriminator(fake_images), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        </pre>
      </div>

      <h4>GAN Challenges</h4>
      <div class="code-block">
        <pre>
# 1. Mode Collapse: G produces limited variety
# Solution: Use techniques like mini-batch discrimination

# 2. Training Instability: D becomes too strong
# Solution: Label smoothing, spectral normalization

# 3. Hard to evaluate: No good metric for quality
# Solution: FID score, Inception Score

# Modern improvements:
# - DCGAN: Use convolutions
# - WGAN: Wasserstein distance for stability
# - StyleGAN: State-of-the-art image quality
# - BigGAN: Large-scale, high-resolution
        </pre>
      </div>

      <h3>Diffusion Models (The Current State-of-the-Art)</h3>
      <p>Diffusion models learn to reverse a gradual noising process.</p>

      <div class="code-block">
        <pre>
# Forward process: Gradually add noise to image
# x_0 (clean) → x_1 → x_2 → ... → x_T (pure noise)

# Reverse process: Learn to denoise step by step
# x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean!)

# Key insight: Denoising is easier than generating from scratch
# Model only needs to remove a little noise at each step
        </pre>
      </div>

      <h4>Forward Diffusion</h4>
      <div class="code-block">
        <pre>
# Add Gaussian noise at each step
# q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

# β_t is a noise schedule (starts small, increases)
# After T steps, x_T ≈ N(0, I) (pure noise)

def forward_diffusion(x_0, t, noise_schedule):
    """Add noise to clean image"""
    alpha_bar = noise_schedule.alpha_bar[t]
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise
        </pre>
      </div>

      <h4>Reverse Diffusion (Denoising)</h4>
      <div class="code-block">
        <pre>
# Train a model to predict the noise that was added
# Given noisy x_t, predict noise ε

class DenoisingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # U-Net architecture with time embedding
        self.time_embed = nn.Embedding(1000, 256)
        # ... encoder, decoder with skip connections

    def forward(self, x_t, t):
        # Predict the noise
        time_emb = self.time_embed(t)
        return self.predict_noise(x_t, time_emb)

# Training:
x_t, noise = forward_diffusion(x_0, t)
predicted_noise = model(x_t, t)
loss = F.mse_loss(predicted_noise, noise)

# Sampling (generation):
x_T = torch.randn(...)  # Start with noise
for t in reversed(range(T)):
    predicted_noise = model(x_t, t)
    x_{t-1} = denoise_step(x_t, predicted_noise, t)
return x_0
        </pre>
      </div>

      <h4>Text-to-Image with Diffusion</h4>
      <div class="code-block">
        <pre>
# DALL-E 2, Stable Diffusion, Midjourney use:
# 1. Text encoder (CLIP) to embed text prompt
# 2. Diffusion in latent space (faster than pixel space)
# 3. Conditioning: Guide denoising toward text embedding

# Classifier-Free Guidance:
# Amplify the difference between conditional and unconditional
output = unconditional + guidance_scale * (conditional - unconditional)

# Higher guidance_scale → more faithful to prompt, less diverse
        </pre>
      </div>

      <h3>Comparison of Generative Models</h3>
      <table>
        <tr><th>Model</th><th>Pros</th><th>Cons</th><th>Use Case</th></tr>
        <tr><td>VAE</td><td>Fast, stable, good latent space</td><td>Blurry outputs</td><td>Compression, interpolation</td></tr>
        <tr><td>GAN</td><td>Sharp images, fast sampling</td><td>Hard to train, mode collapse</td><td>Image synthesis, style transfer</td></tr>
        <tr><td>Diffusion</td><td>Best quality, stable training</td><td>Slow sampling</td><td>Text-to-image, inpainting</td></tr>
      </table>

      <h3>Practical: Using Stable Diffusion</h3>
      <div class="code-block">
        <pre>
from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image from text
prompt = "A serene lake surrounded by mountains at sunset, oil painting"
image = pipe(
    prompt,
    num_inference_steps=50,   # More steps = better quality
    guidance_scale=7.5        # How closely to follow prompt
).images[0]

image.save("generated.png")
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>VAEs</strong> encode to distributions, enabling smooth interpolation</p>
        <p>2. <strong>GANs</strong> use adversarial training - generator vs discriminator</p>
        <p>3. <strong>Diffusion models</strong> learn to reverse a noising process - current SOTA</p>
        <p>4. <strong>Text-to-image</strong> combines text encoders with diffusion models</p>
        <p>5. <strong>Trade-offs:</strong> VAEs are fast but blurry, GANs are sharp but unstable, Diffusion is best quality but slow</p>
      </div>
    `,
    questions: [
      {
        id: "23-1",
        type: "mcq",
        question: "What is the key difference between VAEs and regular autoencoders?",
        options: [
          "VAEs are faster",
          "VAEs encode to a probability distribution, not a fixed point",
          "VAEs don't use neural networks",
          "VAEs can only work with images"
        ],
        correctAnswer: "VAEs encode to a probability distribution, not a fixed point",
        explanation: "Regular autoencoders encode inputs to fixed latent codes, which leads to a discontinuous latent space. VAEs encode to distributions (mean and variance), and the KL divergence loss ensures the latent space is continuous and close to a standard normal - enabling meaningful interpolation and generation."
      },
      {
        id: "23-2",
        type: "mcq",
        question: "In a GAN, what is the role of the discriminator?",
        options: [
          "To generate fake images",
          "To classify real images from fake ones",
          "To compress images",
          "To add noise to images"
        ],
        correctAnswer: "To classify real images from fake ones",
        explanation: "The discriminator's job is to distinguish real images from fake ones generated by the generator. This adversarial setup pushes the generator to create increasingly realistic images to 'fool' the discriminator."
      },
      {
        id: "23-3",
        type: "descriptive",
        question: "Explain how diffusion models generate images and why they produce high-quality results.",
        keywords: ["noise", "denoise", "gradual", "steps", "reverse", "forward", "predict", "stable", "training", "iterative"],
        explanation: "Diffusion models: (1) Forward process gradually adds noise over many steps until the image becomes pure noise. (2) A neural network learns to reverse this - predicting and removing a small amount of noise at each step. (3) Generation starts from random noise and iteratively denoises. Quality is high because: the task of removing small noise is easier than generating from scratch, training is stable (simple MSE loss), and many denoising steps allow fine-grained refinement."
      },
      {
        id: "23-4",
        type: "mcq",
        question: "What is 'mode collapse' in GANs?",
        options: [
          "The discriminator becomes too accurate",
          "The generator produces limited variety, repeating similar outputs",
          "Training takes too long",
          "The model runs out of memory"
        ],
        correctAnswer: "The generator produces limited variety, repeating similar outputs",
        explanation: "Mode collapse occurs when the generator finds a few outputs that fool the discriminator and keeps producing those, ignoring the diversity in the real data. For example, a GAN trained on faces might only generate faces of one gender or ethnicity."
      }
    ]
  }
];

export default premiumLessons;
