export interface Question {
  id: string;
  type: 'mcq' | 'descriptive';
  question: string;
  options?: string[];
  correctAnswer?: string; // For MCQ
  keywords?: string[]; // For descriptive - key concepts that should be mentioned
  explanation: string;
}

export interface Lesson {
  id: number;
  title: string;
  description: string;
  duration: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  content: string; // HTML content
  questions: Question[];
}

export const lessons: Lesson[] = [
  {
    id: 1,
    title: "Introduction to Artificial Intelligence",
    description: "Learn what AI is, its history, and why it matters today",
    duration: "30 min",
    difficulty: "Beginner",
    content: `
      <h2>What is Artificial Intelligence?</h2>
      <p>Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include:</p>
      <ul>
        <li><strong>Learning</strong> - acquiring information and rules for using it</li>
        <li><strong>Reasoning</strong> - using rules to reach approximate or definite conclusions</li>
        <li><strong>Self-correction</strong> - improving performance based on feedback</li>
      </ul>

      <h3>Brief History of AI</h3>
      <div class="timeline">
        <p><strong>1950</strong> - Alan Turing publishes "Computing Machinery and Intelligence" and proposes the Turing Test</p>
        <p><strong>1956</strong> - The term "Artificial Intelligence" is coined at the Dartmouth Conference by John McCarthy</p>
        <p><strong>1966</strong> - ELIZA, an early natural language processing program, is created at MIT</p>
        <p><strong>1997</strong> - IBM's Deep Blue defeats world chess champion Garry Kasparov</p>
        <p><strong>2011</strong> - IBM Watson wins Jeopardy! against human champions</p>
        <p><strong>2012</strong> - Deep learning revolution begins with AlexNet winning ImageNet</p>
        <p><strong>2016</strong> - Google's AlphaGo defeats world Go champion Lee Sedol</p>
        <p><strong>2022-2024</strong> - Large Language Models (ChatGPT, Claude) revolutionize AI accessibility</p>
      </div>

      <h3>Types of AI</h3>
      <p>AI can be categorized based on capabilities:</p>

      <h4>1. Narrow AI (Weak AI)</h4>
      <p>Designed for specific tasks. This is what we have today:</p>
      <ul>
        <li>Voice assistants (Siri, Alexa)</li>
        <li>Recommendation systems (Netflix, Spotify)</li>
        <li>Image recognition</li>
        <li>Language translation</li>
      </ul>

      <h4>2. General AI (Strong AI)</h4>
      <p>Hypothetical AI that matches human cognitive abilities across all domains. Does not exist yet.</p>

      <h4>3. Super AI</h4>
      <p>Theoretical AI that surpasses human intelligence in all aspects. Purely speculative at this point.</p>

      <h3>Why AI Matters Today</h3>
      <p>AI is transforming virtually every industry:</p>
      <ul>
        <li><strong>Healthcare</strong> - Disease diagnosis, drug discovery, personalized medicine</li>
        <li><strong>Finance</strong> - Fraud detection, algorithmic trading, risk assessment</li>
        <li><strong>Transportation</strong> - Self-driving cars, traffic optimization</li>
        <li><strong>Education</strong> - Personalized learning, automated grading</li>
        <li><strong>Entertainment</strong> - Content recommendation, game AI, content generation</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. AI simulates human intelligence through learning, reasoning, and self-correction</p>
        <p>2. We currently have Narrow AI - good at specific tasks but not general intelligence</p>
        <p>3. AI is already transforming major industries and will continue to do so</p>
        <p>4. Understanding AI fundamentals is crucial for participating in the modern economy</p>
      </div>
    `,
    questions: [
      {
        id: "1-1",
        type: "mcq",
        question: "Who coined the term 'Artificial Intelligence'?",
        options: ["Alan Turing", "John McCarthy", "Marvin Minsky", "Geoffrey Hinton"],
        correctAnswer: "John McCarthy",
        explanation: "John McCarthy coined the term 'Artificial Intelligence' at the Dartmouth Conference in 1956, which is considered the birth of AI as a field."
      },
      {
        id: "1-2",
        type: "mcq",
        question: "What type of AI do we currently have in widespread use?",
        options: ["Super AI", "General AI", "Narrow AI", "Conscious AI"],
        correctAnswer: "Narrow AI",
        explanation: "We currently have Narrow AI (also called Weak AI), which is designed for specific tasks like image recognition or language translation. General AI and Super AI remain theoretical."
      },
      {
        id: "1-3",
        type: "descriptive",
        question: "Explain the three main processes that AI systems simulate from human intelligence.",
        keywords: ["learning", "reasoning", "self-correction", "information", "rules", "conclusions", "feedback", "improve"],
        explanation: "AI simulates three main processes: (1) Learning - acquiring information and rules for using it, (2) Reasoning - using rules to reach conclusions, and (3) Self-correction - improving performance based on feedback."
      },
      {
        id: "1-4",
        type: "descriptive",
        question: "Why is AI important today? Give at least 3 examples of how it's being used in different industries.",
        keywords: ["healthcare", "finance", "transportation", "education", "diagnosis", "fraud", "self-driving", "recommendation"],
        explanation: "AI is transforming industries like Healthcare (disease diagnosis, drug discovery), Finance (fraud detection, trading), Transportation (self-driving cars), Education (personalized learning), and Entertainment (recommendations, content generation)."
      }
    ]
  },
  {
    id: 2,
    title: "Machine Learning Fundamentals",
    description: "Understand what machine learning is and how machines learn from data",
    duration: "45 min",
    difficulty: "Beginner",
    content: `
      <h2>What is Machine Learning?</h2>
      <p>Machine Learning (ML) is a subset of AI that enables computers to learn from data without being explicitly programmed. Instead of writing specific rules, we feed data to algorithms and let them discover patterns.</p>

      <div class="highlight">
        <p><strong>Traditional Programming:</strong> Rules + Data → Output</p>
        <p><strong>Machine Learning:</strong> Data + Output → Rules (learned patterns)</p>
      </div>

      <h3>Types of Machine Learning</h3>

      <h4>1. Supervised Learning</h4>
      <p>The algorithm learns from labeled data - examples with known correct answers.</p>
      <p><strong>How it works:</strong></p>
      <ol>
        <li>You provide input data AND the correct outputs (labels)</li>
        <li>The algorithm learns the mapping from inputs to outputs</li>
        <li>It can then predict outputs for new, unseen inputs</li>
      </ol>
      <p><strong>Examples:</strong></p>
      <ul>
        <li>Email spam detection (emails labeled as spam/not spam)</li>
        <li>House price prediction (houses with known prices)</li>
        <li>Image classification (images labeled with categories)</li>
      </ul>
      <p><strong>Common Algorithms:</strong> Linear Regression, Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), Neural Networks</p>

      <h4>2. Unsupervised Learning</h4>
      <p>The algorithm finds patterns in unlabeled data - no correct answers provided.</p>
      <p><strong>How it works:</strong></p>
      <ol>
        <li>You provide only input data (no labels)</li>
        <li>The algorithm discovers hidden patterns or structures</li>
        <li>It groups similar data points together</li>
      </ol>
      <p><strong>Examples:</strong></p>
      <ul>
        <li>Customer segmentation (grouping customers by behavior)</li>
        <li>Anomaly detection (finding unusual patterns)</li>
        <li>Topic modeling (discovering themes in documents)</li>
      </ul>
      <p><strong>Common Algorithms:</strong> K-Means Clustering, Hierarchical Clustering, Principal Component Analysis (PCA), DBSCAN</p>

      <h4>3. Reinforcement Learning</h4>
      <p>The algorithm learns by interacting with an environment and receiving rewards or penalties.</p>
      <p><strong>How it works:</strong></p>
      <ol>
        <li>An agent takes actions in an environment</li>
        <li>It receives rewards (positive) or penalties (negative)</li>
        <li>It learns to maximize cumulative rewards over time</li>
      </ol>
      <p><strong>Examples:</strong></p>
      <ul>
        <li>Game playing (AlphaGo, game bots)</li>
        <li>Robotics (learning to walk, manipulate objects)</li>
        <li>Autonomous vehicles (learning to drive)</li>
      </ul>
      <p><strong>Key Concepts:</strong> Agent, Environment, State, Action, Reward, Policy</p>

      <h3>The Machine Learning Workflow</h3>
      <ol>
        <li><strong>Data Collection</strong> - Gather relevant data for your problem</li>
        <li><strong>Data Preprocessing</strong> - Clean, normalize, and prepare data</li>
        <li><strong>Feature Engineering</strong> - Select and create meaningful features</li>
        <li><strong>Model Selection</strong> - Choose appropriate algorithm(s)</li>
        <li><strong>Training</strong> - Feed data to the model to learn patterns</li>
        <li><strong>Evaluation</strong> - Test model performance on unseen data</li>
        <li><strong>Tuning</strong> - Adjust parameters to improve performance</li>
        <li><strong>Deployment</strong> - Put the model into production</li>
      </ol>

      <h3>Key Concepts</h3>

      <h4>Training vs Testing Data</h4>
      <p>Data is split into:</p>
      <ul>
        <li><strong>Training set (70-80%)</strong> - Used to train the model</li>
        <li><strong>Validation set (10-15%)</strong> - Used to tune hyperparameters</li>
        <li><strong>Test set (10-15%)</strong> - Used to evaluate final performance</li>
      </ul>

      <h4>Overfitting vs Underfitting</h4>
      <ul>
        <li><strong>Overfitting</strong> - Model memorizes training data, performs poorly on new data</li>
        <li><strong>Underfitting</strong> - Model is too simple, fails to capture patterns</li>
        <li><strong>Good fit</strong> - Model generalizes well to new data</li>
      </ul>

      <h4>Features and Labels</h4>
      <ul>
        <li><strong>Features (X)</strong> - Input variables used for prediction</li>
        <li><strong>Labels (Y)</strong> - Output variable we want to predict</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. ML lets computers learn patterns from data instead of explicit programming</p>
        <p>2. Three main types: Supervised (labeled data), Unsupervised (unlabeled), Reinforcement (rewards)</p>
        <p>3. The ML workflow involves data prep, training, evaluation, and deployment</p>
        <p>4. Balance between overfitting and underfitting is crucial</p>
      </div>
    `,
    questions: [
      {
        id: "2-1",
        type: "mcq",
        question: "In supervised learning, what does the algorithm require during training?",
        options: ["Only input data", "Input data and correct outputs (labels)", "Rewards and penalties", "Unlabeled data clusters"],
        correctAnswer: "Input data and correct outputs (labels)",
        explanation: "Supervised learning requires labeled data - both input data AND the correct outputs (labels). The algorithm learns the mapping between inputs and outputs."
      },
      {
        id: "2-2",
        type: "mcq",
        question: "Which type of machine learning is used when AlphaGo learns to play the game of Go?",
        options: ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Transfer Learning"],
        correctAnswer: "Reinforcement Learning",
        explanation: "Reinforcement Learning is used for game playing because the agent (AlphaGo) learns by interacting with the environment (game board) and receiving rewards (winning) or penalties (losing)."
      },
      {
        id: "2-3",
        type: "mcq",
        question: "What is overfitting?",
        options: [
          "When a model is too simple to capture patterns",
          "When a model memorizes training data and performs poorly on new data",
          "When a model has too few features",
          "When training takes too long"
        ],
        correctAnswer: "When a model memorizes training data and performs poorly on new data",
        explanation: "Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, causing it to perform poorly on new, unseen data."
      },
      {
        id: "2-4",
        type: "descriptive",
        question: "Explain the difference between supervised and unsupervised learning. Give one example of each.",
        keywords: ["labeled", "unlabeled", "labels", "patterns", "classification", "clustering", "spam", "customer", "segmentation", "prediction"],
        explanation: "Supervised learning uses labeled data (input + correct output) to learn mappings, like spam detection. Unsupervised learning finds patterns in unlabeled data without correct answers, like customer segmentation."
      },
      {
        id: "2-5",
        type: "descriptive",
        question: "Describe the main steps in a machine learning workflow.",
        keywords: ["data collection", "preprocessing", "feature", "model", "training", "evaluation", "tuning", "deployment", "test"],
        explanation: "ML workflow: 1) Data Collection, 2) Data Preprocessing, 3) Feature Engineering, 4) Model Selection, 5) Training, 6) Evaluation, 7) Tuning, 8) Deployment."
      }
    ]
  },
  {
    id: 3,
    title: "Neural Networks & Deep Learning",
    description: "Dive into how neural networks work and what makes deep learning powerful",
    duration: "60 min",
    difficulty: "Intermediate",
    content: `
      <h2>What are Neural Networks?</h2>
      <p>Neural Networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that process information.</p>

      <h3>The Biological Inspiration</h3>
      <p>In the brain:</p>
      <ul>
        <li>Neurons receive signals through dendrites</li>
        <li>Process signals in the cell body</li>
        <li>Send output through axons to other neurons</li>
        <li>Connections (synapses) have different strengths</li>
      </ul>

      <h3>Artificial Neuron (Perceptron)</h3>
      <p>An artificial neuron mimics this process:</p>
      <ol>
        <li><strong>Inputs (x₁, x₂, ...)</strong> - Like dendrites receiving signals</li>
        <li><strong>Weights (w₁, w₂, ...)</strong> - Strength of each connection</li>
        <li><strong>Sum</strong> - Weighted sum of inputs: Σ(xᵢ × wᵢ) + bias</li>
        <li><strong>Activation Function</strong> - Determines the output</li>
        <li><strong>Output</strong> - The result passed to next layer</li>
      </ol>

      <div class="highlight">
        <p><strong>Neuron Output = Activation(Σ(inputs × weights) + bias)</strong></p>
      </div>

      <h3>Network Architecture</h3>
      <p>Neural networks are organized in layers:</p>

      <h4>1. Input Layer</h4>
      <p>Receives the raw data. Number of neurons equals number of input features.</p>

      <h4>2. Hidden Layers</h4>
      <p>Process information. Can have multiple hidden layers (this is "deep" learning). Each layer extracts increasingly abstract features.</p>

      <h4>3. Output Layer</h4>
      <p>Produces the final result. Structure depends on the task:</p>
      <ul>
        <li>Binary classification: 1 neuron with sigmoid</li>
        <li>Multi-class: Multiple neurons with softmax</li>
        <li>Regression: 1 neuron with linear activation</li>
      </ul>

      <h3>Activation Functions</h3>
      <p>Activation functions introduce non-linearity, allowing networks to learn complex patterns:</p>

      <h4>Common Activation Functions:</h4>
      <ul>
        <li><strong>Sigmoid</strong>: Outputs 0-1, good for probabilities. σ(x) = 1/(1+e⁻ˣ)</li>
        <li><strong>ReLU</strong>: Most popular. f(x) = max(0, x). Fast and effective.</li>
        <li><strong>Tanh</strong>: Outputs -1 to 1. tanh(x)</li>
        <li><strong>Softmax</strong>: For multi-class output. Converts to probability distribution.</li>
        <li><strong>Leaky ReLU</strong>: Fixes "dying ReLU" problem. f(x) = max(0.01x, x)</li>
      </ul>

      <h3>How Neural Networks Learn</h3>

      <h4>1. Forward Propagation</h4>
      <p>Data flows from input to output through the network, producing a prediction.</p>

      <h4>2. Loss Function</h4>
      <p>Measures how wrong the prediction is compared to the actual value:</p>
      <ul>
        <li><strong>MSE (Mean Squared Error)</strong> - For regression</li>
        <li><strong>Cross-Entropy Loss</strong> - For classification</li>
      </ul>

      <h4>3. Backpropagation</h4>
      <p>The key algorithm for training neural networks:</p>
      <ol>
        <li>Calculate the error/loss at the output</li>
        <li>Propagate the error backward through the network</li>
        <li>Calculate gradients (how much each weight contributed to error)</li>
        <li>Update weights to reduce the error</li>
      </ol>

      <h4>4. Gradient Descent</h4>
      <p>Optimization algorithm to minimize loss:</p>
      <ul>
        <li>Calculate gradient (slope) of loss function</li>
        <li>Move weights in opposite direction of gradient</li>
        <li>Learning rate controls step size</li>
        <li>Repeat until convergence</li>
      </ul>
      <p><strong>Weight update rule:</strong> w_new = w_old - learning_rate × gradient</p>

      <h3>What is Deep Learning?</h3>
      <p>Deep Learning is machine learning using neural networks with multiple hidden layers (deep networks). More layers allow learning more complex, hierarchical representations.</p>

      <h4>Why "Deep" Works:</h4>
      <ul>
        <li>Early layers learn simple features (edges, colors)</li>
        <li>Middle layers combine into complex features (shapes, textures)</li>
        <li>Deep layers learn abstract concepts (objects, faces)</li>
      </ul>

      <h3>Common Deep Learning Architectures</h3>

      <h4>Convolutional Neural Networks (CNNs)</h4>
      <p>Specialized for image processing:</p>
      <ul>
        <li>Convolutional layers detect local patterns</li>
        <li>Pooling layers reduce spatial dimensions</li>
        <li>Used for: image classification, object detection, facial recognition</li>
      </ul>

      <h4>Recurrent Neural Networks (RNNs)</h4>
      <p>Designed for sequential data:</p>
      <ul>
        <li>Have memory of previous inputs</li>
        <li>Process sequences step by step</li>
        <li>Used for: language modeling, time series, speech recognition</li>
      </ul>

      <h4>Transformers</h4>
      <p>Modern architecture revolutionizing NLP:</p>
      <ul>
        <li>Use attention mechanisms to weigh importance</li>
        <li>Process all inputs in parallel (faster than RNNs)</li>
        <li>Foundation of GPT, BERT, Claude, and other LLMs</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Neural networks are layers of interconnected neurons that process information</p>
        <p>2. They learn through forward propagation, loss calculation, and backpropagation</p>
        <p>3. Deep learning uses many layers to learn hierarchical representations</p>
        <p>4. Different architectures (CNN, RNN, Transformer) suit different data types</p>
      </div>
    `,
    questions: [
      {
        id: "3-1",
        type: "mcq",
        question: "What is the purpose of an activation function in a neural network?",
        options: [
          "To initialize the weights",
          "To introduce non-linearity so the network can learn complex patterns",
          "To speed up training",
          "To reduce the number of parameters"
        ],
        correctAnswer: "To introduce non-linearity so the network can learn complex patterns",
        explanation: "Activation functions introduce non-linearity into the network. Without them, a neural network would only be able to learn linear relationships, no matter how many layers it has."
      },
      {
        id: "3-2",
        type: "mcq",
        question: "What does backpropagation do?",
        options: [
          "Sends data from input to output",
          "Calculates how much each weight contributed to the error and updates them",
          "Selects the best activation function",
          "Determines the optimal number of layers"
        ],
        correctAnswer: "Calculates how much each weight contributed to the error and updates them",
        explanation: "Backpropagation propagates the error backward through the network, calculating gradients that show how much each weight contributed to the error. These gradients are then used to update the weights."
      },
      {
        id: "3-3",
        type: "mcq",
        question: "Which neural network architecture is most commonly used for image processing?",
        options: ["Recurrent Neural Networks (RNN)", "Transformers", "Convolutional Neural Networks (CNN)", "Autoencoders"],
        correctAnswer: "Convolutional Neural Networks (CNN)",
        explanation: "CNNs are designed specifically for image processing. Their convolutional layers are excellent at detecting local patterns like edges and shapes, making them ideal for image classification and object detection."
      },
      {
        id: "3-4",
        type: "mcq",
        question: "Which activation function is most commonly used in hidden layers of modern neural networks?",
        options: ["Sigmoid", "Tanh", "ReLU", "Softmax"],
        correctAnswer: "ReLU",
        explanation: "ReLU (Rectified Linear Unit) is the most popular activation function for hidden layers because it's computationally efficient and helps avoid the vanishing gradient problem."
      },
      {
        id: "3-5",
        type: "descriptive",
        question: "Explain how a neural network learns. Describe the key steps from input to weight update.",
        keywords: ["forward", "propagation", "loss", "error", "backpropagation", "gradient", "descent", "weights", "update", "prediction"],
        explanation: "Neural networks learn through: 1) Forward propagation - data flows through layers to produce a prediction, 2) Loss calculation - comparing prediction to actual value, 3) Backpropagation - propagating error backward to calculate gradients, 4) Gradient descent - updating weights to minimize loss."
      },
      {
        id: "3-6",
        type: "descriptive",
        question: "What is the difference between CNNs, RNNs, and Transformers? When would you use each?",
        keywords: ["image", "sequential", "attention", "convolutional", "recurrent", "memory", "parallel", "text", "language", "vision"],
        explanation: "CNNs use convolutional layers for image processing (image classification, object detection). RNNs have memory for sequential data (time series, early NLP). Transformers use attention mechanisms for parallel processing (modern NLP, LLMs like GPT and Claude)."
      }
    ]
  },
  {
    id: 4,
    title: "Natural Language Processing (NLP)",
    description: "Learn how AI understands and generates human language",
    duration: "50 min",
    difficulty: "Intermediate",
    content: `
      <h2>What is Natural Language Processing?</h2>
      <p>Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It bridges the gap between human communication and computer understanding.</p>

      <h3>Challenges in NLP</h3>
      <p>Human language is incredibly complex:</p>
      <ul>
        <li><strong>Ambiguity</strong> - "I saw the man with the telescope" (who has the telescope?)</li>
        <li><strong>Context</strong> - "It's cold" could mean temperature or emotional state</li>
        <li><strong>Sarcasm/Irony</strong> - "Oh great, another meeting" (positive words, negative meaning)</li>
        <li><strong>Idioms</strong> - "Break a leg" doesn't mean literally break a leg</li>
        <li><strong>Multiple languages</strong> - Different structures, expressions, cultural context</li>
      </ul>

      <h3>Core NLP Tasks</h3>

      <h4>1. Text Preprocessing</h4>
      <ul>
        <li><strong>Tokenization</strong> - Splitting text into words/subwords</li>
        <li><strong>Lowercasing</strong> - Converting to lowercase for consistency</li>
        <li><strong>Stopword Removal</strong> - Removing common words (the, is, at)</li>
        <li><strong>Stemming</strong> - Reducing words to root form (running → run)</li>
        <li><strong>Lemmatization</strong> - Converting to dictionary form (better → good)</li>
      </ul>

      <h4>2. Text Representation</h4>
      <p>Converting text to numbers that machines can process:</p>

      <p><strong>Bag of Words (BoW)</strong></p>
      <ul>
        <li>Count frequency of each word</li>
        <li>Ignores word order</li>
        <li>Simple but loses context</li>
      </ul>

      <p><strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong></p>
      <ul>
        <li>Weighs words by importance</li>
        <li>Common words get lower weight</li>
        <li>Rare but relevant words get higher weight</li>
      </ul>

      <p><strong>Word Embeddings</strong></p>
      <ul>
        <li>Dense vector representations</li>
        <li>Similar words have similar vectors</li>
        <li>Captures semantic relationships</li>
        <li>Examples: Word2Vec, GloVe, FastText</li>
      </ul>

      <h4>3. Common NLP Applications</h4>

      <p><strong>Sentiment Analysis</strong></p>
      <p>Determining emotional tone of text (positive, negative, neutral)</p>
      <ul>
        <li>Product reviews</li>
        <li>Social media monitoring</li>
        <li>Customer feedback analysis</li>
      </ul>

      <p><strong>Named Entity Recognition (NER)</strong></p>
      <p>Identifying and categorizing entities in text:</p>
      <ul>
        <li>People: "Elon Musk"</li>
        <li>Organizations: "Google"</li>
        <li>Locations: "San Francisco"</li>
        <li>Dates: "January 2024"</li>
      </ul>

      <p><strong>Machine Translation</strong></p>
      <p>Automatically translating between languages (Google Translate, DeepL)</p>

      <p><strong>Question Answering</strong></p>
      <p>Systems that can answer questions based on given context or knowledge</p>

      <p><strong>Text Summarization</strong></p>
      <p>Condensing long documents into shorter versions while preserving key information</p>

      <p><strong>Text Generation</strong></p>
      <p>Creating human-like text (chatbots, content creation, code generation)</p>

      <h3>Evolution of NLP Models</h3>

      <h4>Rule-Based Systems (1950s-1990s)</h4>
      <p>Hand-crafted rules by linguists. Limited scalability.</p>

      <h4>Statistical Methods (1990s-2010s)</h4>
      <p>Probabilistic models learned from data. N-grams, Hidden Markov Models.</p>

      <h4>Word Embeddings (2013+)</h4>
      <p>Word2Vec, GloVe revolutionized representation learning.</p>

      <h4>Recurrent Neural Networks (2014+)</h4>
      <p>LSTMs and GRUs for sequence modeling. Better at capturing context.</p>

      <h4>Transformer Era (2017+)</h4>
      <p>"Attention Is All You Need" paper introduced Transformers.</p>
      <ul>
        <li><strong>BERT (2018)</strong> - Bidirectional understanding, great for classification</li>
        <li><strong>GPT Series (2018+)</strong> - Generative models, text completion</li>
        <li><strong>T5 (2019)</strong> - Text-to-text framework</li>
        <li><strong>GPT-3/4, Claude (2020+)</strong> - Large Language Models with emergent abilities</li>
      </ul>

      <h3>The Transformer Architecture</h3>
      <p>The foundation of modern NLP:</p>

      <h4>Self-Attention Mechanism</h4>
      <p>Allows the model to weigh the importance of different words when processing each word:</p>
      <ul>
        <li>"The cat sat on the mat because it was tired"</li>
        <li>When processing "it", attention helps link it to "cat"</li>
      </ul>

      <h4>Key Components:</h4>
      <ul>
        <li><strong>Multi-Head Attention</strong> - Multiple attention patterns in parallel</li>
        <li><strong>Positional Encoding</strong> - Adds word position information</li>
        <li><strong>Feed-Forward Networks</strong> - Process attention outputs</li>
        <li><strong>Layer Normalization</strong> - Stabilizes training</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. NLP enables computers to understand and generate human language</p>
        <p>2. Text must be converted to numbers through tokenization and embeddings</p>
        <p>3. Common tasks include sentiment analysis, NER, translation, and generation</p>
        <p>4. Transformers and attention mechanisms are the foundation of modern NLP</p>
      </div>
    `,
    questions: [
      {
        id: "4-1",
        type: "mcq",
        question: "What is tokenization in NLP?",
        options: [
          "Converting text to uppercase",
          "Splitting text into smaller units like words or subwords",
          "Translating text to another language",
          "Removing all punctuation"
        ],
        correctAnswer: "Splitting text into smaller units like words or subwords",
        explanation: "Tokenization is the process of breaking down text into smaller units (tokens), typically words or subwords. This is a fundamental preprocessing step in NLP."
      },
      {
        id: "4-2",
        type: "mcq",
        question: "What advantage do word embeddings have over Bag of Words?",
        options: [
          "They are faster to compute",
          "They capture semantic relationships between words",
          "They require less memory",
          "They work with any language automatically"
        ],
        correctAnswer: "They capture semantic relationships between words",
        explanation: "Word embeddings represent words as dense vectors where similar words have similar vectors. This captures semantic relationships (e.g., 'king' - 'man' + 'woman' ≈ 'queen'), unlike Bag of Words which just counts occurrences."
      },
      {
        id: "4-3",
        type: "mcq",
        question: "What is the key innovation in Transformer architecture?",
        options: [
          "Convolutional layers",
          "Recurrent connections",
          "Self-attention mechanism",
          "Pooling layers"
        ],
        correctAnswer: "Self-attention mechanism",
        explanation: "The self-attention mechanism is the key innovation in Transformers. It allows the model to weigh the importance of different parts of the input when processing each element, enabling parallel processing and better long-range dependencies."
      },
      {
        id: "4-4",
        type: "descriptive",
        question: "What is sentiment analysis and give 2 examples of where it's used?",
        keywords: ["emotion", "tone", "positive", "negative", "review", "social media", "feedback", "customer", "opinion"],
        explanation: "Sentiment analysis determines the emotional tone of text (positive, negative, neutral). It's used in analyzing product reviews, social media monitoring, and customer feedback analysis to understand public opinion."
      },
      {
        id: "4-5",
        type: "descriptive",
        question: "Explain the evolution of NLP from rule-based systems to Transformers.",
        keywords: ["rule", "statistical", "embedding", "word2vec", "RNN", "LSTM", "transformer", "attention", "BERT", "GPT"],
        explanation: "NLP evolved from: 1) Rule-based systems (hand-crafted rules), 2) Statistical methods (probabilistic models), 3) Word embeddings (Word2Vec, GloVe), 4) RNNs/LSTMs (sequence modeling), 5) Transformers (attention-based, parallel processing) which power modern LLMs like GPT and Claude."
      }
    ]
  },
  {
    id: 5,
    title: "Computer Vision",
    description: "Explore how AI systems see and understand images and video",
    duration: "45 min",
    difficulty: "Intermediate",
    content: `
      <h2>What is Computer Vision?</h2>
      <p>Computer Vision is a field of AI that enables machines to interpret and understand visual information from the world - images, videos, and real-time camera feeds.</p>

      <h3>How Computers "See" Images</h3>
      <p>To a computer, an image is just a matrix of numbers:</p>
      <ul>
        <li><strong>Grayscale</strong> - 2D matrix, each cell is 0-255 (black to white)</li>
        <li><strong>Color (RGB)</strong> - 3 stacked matrices (Red, Green, Blue channels)</li>
        <li><strong>Resolution</strong> - Width × Height × Channels (e.g., 224×224×3)</li>
      </ul>

      <h3>Core Computer Vision Tasks</h3>

      <h4>1. Image Classification</h4>
      <p>Assigning a label to an entire image.</p>
      <ul>
        <li>Input: An image</li>
        <li>Output: Category label (e.g., "cat", "dog", "car")</li>
        <li>Example: Is this image a dog or a cat?</li>
      </ul>

      <h4>2. Object Detection</h4>
      <p>Finding and locating objects within an image.</p>
      <ul>
        <li>Input: An image</li>
        <li>Output: Bounding boxes + labels for each object</li>
        <li>Example: Find all cars in a street scene</li>
      </ul>

      <h4>3. Semantic Segmentation</h4>
      <p>Classifying every pixel in an image.</p>
      <ul>
        <li>Input: An image</li>
        <li>Output: Label for each pixel</li>
        <li>Example: Medical imaging, autonomous driving scene understanding</li>
      </ul>

      <h4>4. Instance Segmentation</h4>
      <p>Segmentation that distinguishes individual objects.</p>
      <ul>
        <li>Combines object detection + segmentation</li>
        <li>Can tell apart individual instances (person 1, person 2)</li>
      </ul>

      <h4>5. Pose Estimation</h4>
      <p>Detecting human body positions and poses.</p>
      <ul>
        <li>Identifies key points (joints, facial features)</li>
        <li>Used in sports analysis, gaming, AR</li>
      </ul>

      <h4>6. Image Generation</h4>
      <p>Creating new images from scratch or modifying existing ones.</p>
      <ul>
        <li>GANs (Generative Adversarial Networks)</li>
        <li>Diffusion Models (Stable Diffusion, DALL-E, Midjourney)</li>
      </ul>

      <h3>Convolutional Neural Networks (CNNs) Deep Dive</h3>
      <p>CNNs are the backbone of modern computer vision:</p>

      <h4>Convolutional Layer</h4>
      <p>The core building block:</p>
      <ul>
        <li><strong>Kernel/Filter</strong> - Small matrix (e.g., 3×3) that slides over the image</li>
        <li><strong>Feature Maps</strong> - Output showing where features are detected</li>
        <li><strong>Stride</strong> - How many pixels the kernel moves each step</li>
        <li><strong>Padding</strong> - Adding zeros around edges to control output size</li>
      </ul>

      <h4>What Convolutions Learn:</h4>
      <ul>
        <li><strong>Early layers</strong> - Edges, colors, simple textures</li>
        <li><strong>Middle layers</strong> - Shapes, patterns, parts of objects</li>
        <li><strong>Deep layers</strong> - Complex features, entire objects</li>
      </ul>

      <h4>Pooling Layer</h4>
      <p>Reduces spatial dimensions while keeping important features:</p>
      <ul>
        <li><strong>Max Pooling</strong> - Takes maximum value in each region</li>
        <li><strong>Average Pooling</strong> - Takes average value</li>
        <li>Reduces computation and adds translation invariance</li>
      </ul>

      <h4>Fully Connected Layer</h4>
      <p>Final layers that make the classification decision after feature extraction.</p>

      <h3>Famous CNN Architectures</h3>

      <h4>LeNet-5 (1998)</h4>
      <p>Pioneer CNN by Yann LeCun for digit recognition.</p>

      <h4>AlexNet (2012)</h4>
      <p>Won ImageNet, sparked deep learning revolution. Used ReLU, dropout, GPU training.</p>

      <h4>VGGNet (2014)</h4>
      <p>Showed depth matters. Used small 3×3 filters throughout.</p>

      <h4>ResNet (2015)</h4>
      <p>Introduced skip connections, enabling very deep networks (152+ layers).</p>

      <h4>EfficientNet (2019)</h4>
      <p>Balanced scaling of depth, width, and resolution for efficiency.</p>

      <h3>Modern Vision Models</h3>

      <h4>Vision Transformers (ViT)</h4>
      <p>Applies Transformer architecture to images:</p>
      <ul>
        <li>Splits image into patches</li>
        <li>Treats patches like words in NLP</li>
        <li>Uses self-attention for global understanding</li>
      </ul>

      <h4>CLIP (Contrastive Language-Image Pre-training)</h4>
      <p>Connects images and text:</p>
      <ul>
        <li>Trained on image-text pairs from internet</li>
        <li>Can classify images using natural language descriptions</li>
        <li>Zero-shot capabilities</li>
      </ul>

      <h3>Real-World Applications</h3>
      <ul>
        <li><strong>Autonomous Vehicles</strong> - Object detection, lane detection, pedestrian tracking</li>
        <li><strong>Medical Imaging</strong> - Tumor detection, X-ray analysis, pathology</li>
        <li><strong>Facial Recognition</strong> - Security, authentication, photo organization</li>
        <li><strong>Retail</strong> - Visual search, inventory management, checkout-free stores</li>
        <li><strong>Manufacturing</strong> - Quality control, defect detection</li>
        <li><strong>Agriculture</strong> - Crop health monitoring, pest detection</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Images are matrices of numbers that CNNs process through convolutions</p>
        <p>2. Tasks range from classification to detection to segmentation to generation</p>
        <p>3. CNNs learn hierarchical features from simple edges to complex objects</p>
        <p>4. Vision Transformers are emerging as powerful alternatives to CNNs</p>
      </div>
    `,
    questions: [
      {
        id: "5-1",
        type: "mcq",
        question: "What is the difference between image classification and object detection?",
        options: [
          "Classification is faster than detection",
          "Classification labels the whole image; detection finds and locates multiple objects",
          "Detection only works on videos",
          "Classification requires more data"
        ],
        correctAnswer: "Classification labels the whole image; detection finds and locates multiple objects",
        explanation: "Image classification assigns one label to the entire image (e.g., 'cat'). Object detection finds multiple objects within an image and provides both labels AND their locations (bounding boxes)."
      },
      {
        id: "5-2",
        type: "mcq",
        question: "What does a convolutional layer detect?",
        options: [
          "The color of the image",
          "Features/patterns in the image using learnable filters",
          "The size of the image",
          "The file format"
        ],
        correctAnswer: "Features/patterns in the image using learnable filters",
        explanation: "Convolutional layers use learnable filters (kernels) to detect features and patterns in images. Early layers detect simple features like edges, while deeper layers detect complex patterns."
      },
      {
        id: "5-3",
        type: "mcq",
        question: "What innovation did ResNet introduce?",
        options: [
          "Dropout regularization",
          "Batch normalization",
          "Skip/residual connections",
          "Attention mechanism"
        ],
        correctAnswer: "Skip/residual connections",
        explanation: "ResNet introduced skip (residual) connections that allow gradients to flow directly through the network, enabling training of very deep networks (100+ layers) without degradation."
      },
      {
        id: "5-4",
        type: "descriptive",
        question: "Explain what a CNN learns at different layer depths (early, middle, deep layers).",
        keywords: ["edge", "color", "shape", "texture", "pattern", "object", "complex", "simple", "feature", "hierarchical"],
        explanation: "CNNs learn hierarchically: Early layers detect simple features (edges, colors, basic textures). Middle layers combine these into shapes, patterns, and object parts. Deep layers recognize complex features and entire objects."
      },
      {
        id: "5-5",
        type: "descriptive",
        question: "Name 3 real-world applications of computer vision and explain how they use it.",
        keywords: ["autonomous", "medical", "facial", "retail", "manufacturing", "detection", "recognition", "analysis", "vehicle", "security"],
        explanation: "1) Autonomous vehicles use object detection for pedestrians/cars and lane detection. 2) Medical imaging uses classification/segmentation for tumor detection. 3) Facial recognition uses detection and embedding matching for security/authentication."
      }
    ]
  },
  {
    id: 6,
    title: "Large Language Models (LLMs)",
    description: "Understand how GPT, Claude, and other LLMs work",
    duration: "60 min",
    difficulty: "Advanced",
    content: `
      <h2>What are Large Language Models?</h2>
      <p>Large Language Models (LLMs) are AI systems trained on massive amounts of text data to understand and generate human-like language. They're the technology behind ChatGPT, Claude, and many other AI assistants.</p>

      <h3>What Makes LLMs "Large"?</h3>
      <ul>
        <li><strong>Parameters</strong> - Billions of learnable weights (GPT-3: 175B, GPT-4: ~1T estimated)</li>
        <li><strong>Training Data</strong> - Hundreds of billions of tokens from books, websites, code</li>
        <li><strong>Compute</strong> - Thousands of GPUs training for weeks/months</li>
      </ul>

      <h3>How LLMs Work</h3>

      <h4>1. Tokenization</h4>
      <p>Text is broken into tokens (subword units):</p>
      <ul>
        <li>"Hello world" → ["Hello", " world"]</li>
        <li>"unbelievable" → ["un", "believ", "able"]</li>
        <li>Most LLMs use ~50,000-100,000 token vocabulary</li>
      </ul>

      <h4>2. Next Token Prediction</h4>
      <p>The core training objective:</p>
      <div class="highlight">
        <p>Given: "The cat sat on the"</p>
        <p>Predict: "mat" (or other likely words)</p>
      </div>
      <p>By predicting the next token billions of times, the model learns language patterns, facts, and reasoning.</p>

      <h4>3. Transformer Architecture</h4>
      <p>LLMs are built on the Transformer architecture:</p>
      <ul>
        <li><strong>Self-Attention</strong> - Lets each token attend to all other tokens</li>
        <li><strong>Feed-Forward Networks</strong> - Process attention outputs</li>
        <li><strong>Layer Stacking</strong> - Many transformer blocks (96 for GPT-3)</li>
        <li><strong>Residual Connections</strong> - Help with training deep networks</li>
      </ul>

      <h4>4. The Attention Mechanism</h4>
      <p>How the model focuses on relevant context:</p>
      <ul>
        <li><strong>Query (Q)</strong> - What am I looking for?</li>
        <li><strong>Key (K)</strong> - What do I have to offer?</li>
        <li><strong>Value (V)</strong> - What information do I contain?</li>
      </ul>
      <p>Attention(Q, K, V) = softmax(QK^T / √d) × V</p>

      <h3>Training LLMs</h3>

      <h4>Pre-training</h4>
      <p>Learning from raw text (self-supervised):</p>
      <ol>
        <li>Collect massive text corpus (internet, books, code)</li>
        <li>Train on next-token prediction</li>
        <li>Model learns grammar, facts, reasoning patterns</li>
        <li>This is the most expensive phase (millions of dollars)</li>
      </ol>

      <h4>Fine-tuning</h4>
      <p>Adapting for specific tasks:</p>
      <ul>
        <li><strong>Supervised Fine-tuning (SFT)</strong> - Train on curated examples</li>
        <li><strong>RLHF (Reinforcement Learning from Human Feedback)</strong> - Learn from human preferences</li>
      </ul>

      <h4>RLHF Process:</h4>
      <ol>
        <li>Generate multiple responses to a prompt</li>
        <li>Humans rank responses by quality</li>
        <li>Train a reward model on these preferences</li>
        <li>Use RL to optimize the LLM against the reward model</li>
      </ol>

      <h3>Emergent Abilities</h3>
      <p>Capabilities that appear only at scale:</p>
      <ul>
        <li><strong>In-context Learning</strong> - Learning from examples in the prompt</li>
        <li><strong>Chain-of-Thought Reasoning</strong> - Step-by-step problem solving</li>
        <li><strong>Code Generation</strong> - Writing functional programs</li>
        <li><strong>Multi-step Reasoning</strong> - Complex logical deductions</li>
      </ul>

      <h3>Prompting Techniques</h3>

      <h4>Zero-shot Prompting</h4>
      <p>Just describe the task:</p>
      <div class="code-block">
        Translate to French: "Hello, how are you?"
      </div>

      <h4>Few-shot Prompting</h4>
      <p>Provide examples:</p>
      <div class="code-block">
        English: Hello → French: Bonjour
        English: Goodbye → French: Au revoir
        English: Thank you → French: ?
      </div>

      <h4>Chain-of-Thought (CoT)</h4>
      <p>Encourage step-by-step reasoning:</p>
      <div class="code-block">
        Q: If there are 3 cars and each has 4 wheels, how many wheels total?
        A: Let's think step by step. Each car has 4 wheels.
           With 3 cars: 3 × 4 = 12 wheels total.
      </div>

      <h4>System Prompts</h4>
      <p>Set behavior and personality:</p>
      <div class="code-block">
        You are a helpful coding assistant. You write clean,
        well-documented code and explain your reasoning.
      </div>

      <h3>Limitations of LLMs</h3>

      <h4>Hallucinations</h4>
      <p>Generating plausible but false information confidently.</p>

      <h4>Context Window</h4>
      <p>Limited memory (though expanding: GPT-4 has 128K tokens, Claude has 200K).</p>

      <h4>Training Data Cutoff</h4>
      <p>Knowledge is frozen at training time.</p>

      <h4>Reasoning Limitations</h4>
      <p>Can struggle with complex math, logic puzzles, novel situations.</p>

      <h4>Bias</h4>
      <p>Can reflect biases present in training data.</p>

      <h3>Popular LLMs</h3>
      <table class="comparison-table">
        <tr><th>Model</th><th>Company</th><th>Notes</th></tr>
        <tr><td>GPT-4</td><td>OpenAI</td><td>Multimodal, strong reasoning</td></tr>
        <tr><td>Claude 3</td><td>Anthropic</td><td>Long context, safety-focused</td></tr>
        <tr><td>Gemini</td><td>Google</td><td>Multimodal, integrated with Google</td></tr>
        <tr><td>LLaMA 3</td><td>Meta</td><td>Open weights, customizable</td></tr>
        <tr><td>Mistral</td><td>Mistral AI</td><td>Efficient, open models</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. LLMs are trained on next-token prediction using massive text data</p>
        <p>2. Transformers with self-attention are the core architecture</p>
        <p>3. RLHF aligns models with human preferences</p>
        <p>4. Prompting techniques like few-shot and CoT improve performance</p>
        <p>5. LLMs have limitations: hallucinations, context limits, training cutoff</p>
      </div>
    `,
    questions: [
      {
        id: "6-1",
        type: "mcq",
        question: "What is the core training objective of most LLMs?",
        options: [
          "Image classification",
          "Next token prediction",
          "Sentiment analysis",
          "Speech recognition"
        ],
        correctAnswer: "Next token prediction",
        explanation: "LLMs are primarily trained on next token prediction - given a sequence of tokens, predict the most likely next token. This simple objective, applied at scale, leads to sophisticated language understanding."
      },
      {
        id: "6-2",
        type: "mcq",
        question: "What does RLHF stand for and what is its purpose?",
        options: [
          "Recursive Learning for Human Features - for feature extraction",
          "Reinforcement Learning from Human Feedback - to align models with human preferences",
          "Rapid Learning through Hybrid Functions - to speed up training",
          "Reasoning Logic for Hierarchical Functions - for logical reasoning"
        ],
        correctAnswer: "Reinforcement Learning from Human Feedback - to align models with human preferences",
        explanation: "RLHF (Reinforcement Learning from Human Feedback) is used to align LLMs with human preferences. Humans rank model outputs, a reward model learns these preferences, and RL optimizes the LLM to produce better responses."
      },
      {
        id: "6-3",
        type: "mcq",
        question: "What is a 'hallucination' in the context of LLMs?",
        options: [
          "When the model generates images",
          "When the model generates plausible but false information",
          "When the model stops responding",
          "When the model repeats itself"
        ],
        correctAnswer: "When the model generates plausible but false information",
        explanation: "Hallucination refers to LLMs generating information that sounds plausible and confident but is actually false or made up. This is a significant challenge because the false information can be hard to detect."
      },
      {
        id: "6-4",
        type: "mcq",
        question: "What is Chain-of-Thought (CoT) prompting?",
        options: [
          "Connecting multiple LLMs together",
          "Asking the model to think step-by-step before answering",
          "Training the model on sequential data",
          "Using multiple prompts in sequence"
        ],
        correctAnswer: "Asking the model to think step-by-step before answering",
        explanation: "Chain-of-Thought prompting encourages the model to break down complex problems into steps, showing its reasoning process. This often improves accuracy on math, logic, and reasoning tasks."
      },
      {
        id: "6-5",
        type: "descriptive",
        question: "Explain the two main phases of training an LLM (pre-training and fine-tuning).",
        keywords: ["pre-training", "fine-tuning", "text", "corpus", "next token", "supervised", "RLHF", "human", "feedback", "examples"],
        explanation: "Pre-training: LLM learns from massive text corpus through next-token prediction, acquiring language patterns and knowledge. Fine-tuning: Model is adapted using supervised examples (SFT) and/or RLHF to follow instructions and align with human preferences."
      },
      {
        id: "6-6",
        type: "descriptive",
        question: "What are 3 limitations of current LLMs?",
        keywords: ["hallucination", "context", "window", "cutoff", "bias", "reasoning", "false", "memory", "knowledge", "limit"],
        explanation: "Key limitations: 1) Hallucinations - generating false but plausible information, 2) Context window limits - can only process limited text at once, 3) Training data cutoff - knowledge is frozen at training time, 4) Reasoning limitations - can struggle with complex logic, 5) Bias from training data."
      }
    ]
  },
  {
    id: 7,
    title: "Building AI Applications",
    description: "Learn how to integrate AI into real-world applications",
    duration: "50 min",
    difficulty: "Advanced",
    content: `
      <h2>From Models to Applications</h2>
      <p>Understanding AI concepts is one thing; building useful applications is another. This lesson covers how to turn AI models into real products.</p>

      <h3>AI Application Architecture</h3>

      <h4>Components of an AI Application:</h4>
      <ul>
        <li><strong>User Interface</strong> - How users interact (web, mobile, API)</li>
        <li><strong>Application Logic</strong> - Business rules, orchestration</li>
        <li><strong>AI/ML Layer</strong> - Models, inference, processing</li>
        <li><strong>Data Layer</strong> - Storage, retrieval, caching</li>
        <li><strong>Infrastructure</strong> - Servers, GPUs, scaling</li>
      </ul>

      <h3>Using AI APIs</h3>
      <p>The fastest way to add AI to applications:</p>

      <h4>Popular AI APIs:</h4>
      <ul>
        <li><strong>OpenAI API</strong> - GPT-4, DALL-E, Whisper</li>
        <li><strong>Anthropic API</strong> - Claude models</li>
        <li><strong>Google AI</strong> - Gemini, PaLM, Vision AI</li>
        <li><strong>Hugging Face</strong> - Thousands of open models</li>
        <li><strong>Replicate</strong> - Run open-source models easily</li>
      </ul>

      <h4>Example: Calling Claude API</h4>
      <div class="code-block">
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic();

const message = await client.messages.create({
  model: "claude-3-opus-20240229",
  max_tokens: 1024,
  messages: [
    { role: "user", content: "Explain quantum computing in simple terms" }
  ]
});

console.log(message.content);
      </div>

      <h3>Prompt Engineering</h3>
      <p>Crafting effective prompts is crucial:</p>

      <h4>Best Practices:</h4>
      <ul>
        <li><strong>Be specific</strong> - Clear instructions produce better results</li>
        <li><strong>Provide context</strong> - Background information helps</li>
        <li><strong>Use examples</strong> - Show the format you want</li>
        <li><strong>Set constraints</strong> - Length, format, tone</li>
        <li><strong>Iterate</strong> - Test and refine prompts</li>
      </ul>

      <h4>Structured Prompts:</h4>
      <div class="code-block">
System: You are a helpful assistant that writes Python code.
Always include comments and handle errors gracefully.

User: Write a function to fetch weather data from an API.

Expected output format:
- Function with type hints
- Error handling with try/except
- Brief docstring explaining usage
      </div>

      <h3>Retrieval-Augmented Generation (RAG)</h3>
      <p>Combining LLMs with external knowledge:</p>

      <h4>How RAG Works:</h4>
      <ol>
        <li><strong>Index</strong> - Convert documents to embeddings, store in vector database</li>
        <li><strong>Retrieve</strong> - Find relevant documents for user query</li>
        <li><strong>Generate</strong> - Include retrieved context in LLM prompt</li>
      </ol>

      <h4>RAG Architecture:</h4>
      <div class="code-block">
User Query
    ↓
Embedding Model (convert to vector)
    ↓
Vector Database (find similar documents)
    ↓
Retrieved Context + Original Query
    ↓
LLM (generate answer with context)
    ↓
Response
      </div>

      <h4>Vector Databases:</h4>
      <ul>
        <li><strong>Pinecone</strong> - Managed, scalable</li>
        <li><strong>Weaviate</strong> - Open-source, feature-rich</li>
        <li><strong>Chroma</strong> - Lightweight, developer-friendly</li>
        <li><strong>pgvector</strong> - PostgreSQL extension</li>
      </ul>

      <h3>Fine-tuning Models</h3>
      <p>When to fine-tune vs use prompting:</p>

      <h4>Use Prompting When:</h4>
      <ul>
        <li>Task can be described in instructions</li>
        <li>You have few examples</li>
        <li>You need flexibility</li>
        <li>Quick iteration is important</li>
      </ul>

      <h4>Fine-tune When:</h4>
      <ul>
        <li>You have lots of task-specific data</li>
        <li>Consistent output format is critical</li>
        <li>You need to reduce latency/costs</li>
        <li>Domain-specific terminology is important</li>
      </ul>

      <h3>AI Agents</h3>
      <p>Systems that can take actions, not just generate text:</p>

      <h4>Agent Components:</h4>
      <ul>
        <li><strong>LLM</strong> - The "brain" for reasoning</li>
        <li><strong>Tools</strong> - Functions the agent can call (search, calculator, APIs)</li>
        <li><strong>Memory</strong> - Storing conversation history and learnings</li>
        <li><strong>Planning</strong> - Breaking down complex tasks</li>
      </ul>

      <h4>Agent Frameworks:</h4>
      <ul>
        <li><strong>LangChain</strong> - Popular, comprehensive</li>
        <li><strong>AutoGPT</strong> - Autonomous task execution</li>
        <li><strong>CrewAI</strong> - Multi-agent collaboration</li>
        <li><strong>Claude Computer Use</strong> - Browser/computer automation</li>
      </ul>

      <h3>Evaluation and Testing</h3>

      <h4>Metrics for AI Applications:</h4>
      <ul>
        <li><strong>Accuracy</strong> - Is the output correct?</li>
        <li><strong>Relevance</strong> - Does it address the query?</li>
        <li><strong>Coherence</strong> - Is it well-structured and logical?</li>
        <li><strong>Latency</strong> - How fast is the response?</li>
        <li><strong>Cost</strong> - Token usage, API costs</li>
      </ul>

      <h4>Testing Strategies:</h4>
      <ul>
        <li>Create test datasets with expected outputs</li>
        <li>Use LLMs to evaluate LLM outputs</li>
        <li>Human evaluation for quality assessment</li>
        <li>A/B testing in production</li>
      </ul>

      <h3>Deployment Considerations</h3>

      <h4>Scalability:</h4>
      <ul>
        <li>Use queuing for heavy workloads</li>
        <li>Cache common responses</li>
        <li>Consider serverless for variable traffic</li>
      </ul>

      <h4>Cost Optimization:</h4>
      <ul>
        <li>Use smaller models when possible</li>
        <li>Implement token budgets</li>
        <li>Cache and reuse embeddings</li>
        <li>Batch requests when possible</li>
      </ul>

      <h4>Safety:</h4>
      <ul>
        <li>Input validation and sanitization</li>
        <li>Output filtering for harmful content</li>
        <li>Rate limiting</li>
        <li>User authentication</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. AI APIs provide the fastest path to adding AI capabilities</p>
        <p>2. RAG extends LLMs with external knowledge</p>
        <p>3. Agents can take actions using tools, not just generate text</p>
        <p>4. Evaluation should cover accuracy, relevance, latency, and cost</p>
        <p>5. Production systems need attention to scaling, costs, and safety</p>
      </div>
    `,
    questions: [
      {
        id: "7-1",
        type: "mcq",
        question: "What is RAG (Retrieval-Augmented Generation)?",
        options: [
          "A way to train models faster",
          "Combining LLMs with external knowledge retrieval",
          "A type of neural network",
          "A method for compressing models"
        ],
        correctAnswer: "Combining LLMs with external knowledge retrieval",
        explanation: "RAG combines LLMs with retrieval systems. Documents are converted to embeddings, stored in a vector database, and relevant ones are retrieved to provide context for the LLM when answering queries."
      },
      {
        id: "7-2",
        type: "mcq",
        question: "When should you fine-tune a model instead of using prompting?",
        options: [
          "When you have very few examples",
          "When you need quick iteration",
          "When you have lots of task-specific data and need consistent outputs",
          "When the task is simple"
        ],
        correctAnswer: "When you have lots of task-specific data and need consistent outputs",
        explanation: "Fine-tuning is best when you have abundant task-specific data, need consistent output formats, want to reduce costs/latency, or need domain-specific knowledge. Prompting is better for flexibility and quick iteration."
      },
      {
        id: "7-3",
        type: "mcq",
        question: "What are the main components of an AI agent?",
        options: [
          "Only an LLM",
          "LLM, tools, memory, and planning capabilities",
          "Just a database and API",
          "User interface and backend"
        ],
        correctAnswer: "LLM, tools, memory, and planning capabilities",
        explanation: "AI agents consist of: an LLM (the 'brain'), tools (functions it can call), memory (for conversation history), and planning (for breaking down complex tasks)."
      },
      {
        id: "7-4",
        type: "descriptive",
        question: "Explain how RAG works step by step.",
        keywords: ["embed", "vector", "database", "retrieve", "query", "context", "document", "generate", "relevant", "store"],
        explanation: "RAG works by: 1) Indexing - converting documents to embeddings and storing in a vector database, 2) Retrieval - when a query comes, finding relevant documents via similarity search, 3) Generation - including retrieved context in the LLM prompt to generate an informed answer."
      },
      {
        id: "7-5",
        type: "descriptive",
        question: "What are important considerations when deploying AI applications to production?",
        keywords: ["scale", "cost", "safety", "cache", "rate limit", "latency", "token", "filter", "validation", "queue"],
        explanation: "Production considerations include: Scalability (queuing, caching, serverless), Cost optimization (smaller models, token budgets, caching embeddings), and Safety (input validation, output filtering, rate limiting, authentication)."
      }
    ]
  },
  {
    id: 8,
    title: "Ethics and Safety in AI",
    description: "Understand the ethical considerations and safety challenges in AI development",
    duration: "40 min",
    difficulty: "Intermediate",
    content: `
      <h2>Why AI Ethics Matter</h2>
      <p>As AI becomes more powerful and ubiquitous, the ethical implications become increasingly important. Responsible AI development isn't just about what we CAN build, but what we SHOULD build.</p>

      <h3>Key Ethical Concerns</h3>

      <h4>1. Bias and Fairness</h4>
      <p>AI systems can perpetuate or amplify existing biases:</p>
      <ul>
        <li><strong>Training data bias</strong> - Models learn biases present in data</li>
        <li><strong>Representation bias</strong> - Underrepresentation of certain groups</li>
        <li><strong>Historical bias</strong> - Past discrimination encoded in data</li>
      </ul>
      <p><strong>Examples:</strong></p>
      <ul>
        <li>Facial recognition performing worse on darker skin tones</li>
        <li>Resume screening systems discriminating against women</li>
        <li>Language models reflecting gender stereotypes</li>
      </ul>

      <h4>2. Privacy</h4>
      <p>AI raises significant privacy concerns:</p>
      <ul>
        <li><strong>Data collection</strong> - Training requires vast amounts of data</li>
        <li><strong>Surveillance</strong> - Facial recognition, behavior tracking</li>
        <li><strong>Data leakage</strong> - Models can memorize and leak training data</li>
        <li><strong>Inference</strong> - AI can infer sensitive information from innocuous data</li>
      </ul>

      <h4>3. Transparency and Explainability</h4>
      <p>The "black box" problem:</p>
      <ul>
        <li>Deep learning models are often unexplainable</li>
        <li>Hard to understand why a specific decision was made</li>
        <li>Crucial in high-stakes domains (healthcare, criminal justice)</li>
      </ul>

      <h4>4. Accountability</h4>
      <p>Who is responsible when AI causes harm?</p>
      <ul>
        <li>Developers who built the system?</li>
        <li>Companies that deployed it?</li>
        <li>Users who operated it?</li>
        <li>The AI itself?</li>
      </ul>

      <h4>5. Job Displacement</h4>
      <p>AI automation impacts employment:</p>
      <ul>
        <li>Some jobs will be eliminated</li>
        <li>New jobs will be created</li>
        <li>Need for reskilling and social safety nets</li>
        <li>Questions about wealth distribution</li>
      </ul>

      <h3>AI Safety</h3>
      <p>Ensuring AI systems behave as intended:</p>

      <h4>Current Safety Challenges:</h4>
      <ul>
        <li><strong>Misuse</strong> - AI used for harmful purposes (deepfakes, scams)</li>
        <li><strong>Accidents</strong> - Unintended behaviors causing harm</li>
        <li><strong>Prompt injection</strong> - Manipulating AI through crafted inputs</li>
        <li><strong>Jailbreaking</strong> - Bypassing safety guardrails</li>
      </ul>

      <h4>Long-term Safety Concerns:</h4>
      <ul>
        <li><strong>Alignment</strong> - Ensuring AI goals match human values</li>
        <li><strong>Control</strong> - Maintaining human oversight as AI becomes more capable</li>
        <li><strong>Existential risk</strong> - Potential risks from superintelligent AI</li>
      </ul>

      <h3>Alignment Problem</h3>
      <p>Making AI systems that reliably do what we want:</p>

      <h4>Challenges:</h4>
      <ul>
        <li><strong>Specification</strong> - Precisely defining what we want is hard</li>
        <li><strong>Reward hacking</strong> - AI finds unexpected ways to maximize reward</li>
        <li><strong>Goal misgeneralization</strong> - AI learns wrong patterns</li>
        <li><strong>Deception</strong> - AI could learn to deceive evaluators</li>
      </ul>

      <h4>Alignment Approaches:</h4>
      <ul>
        <li><strong>RLHF</strong> - Learning from human preferences</li>
        <li><strong>Constitutional AI</strong> - Training AI to follow principles</li>
        <li><strong>Debate</strong> - AI systems arguing to find truth</li>
        <li><strong>Interpretability</strong> - Understanding what models are "thinking"</li>
      </ul>

      <h3>Responsible AI Principles</h3>

      <h4>Key Principles:</h4>
      <ul>
        <li><strong>Fairness</strong> - Equal treatment regardless of demographics</li>
        <li><strong>Transparency</strong> - Clear about AI use and limitations</li>
        <li><strong>Privacy</strong> - Protect user data and respect consent</li>
        <li><strong>Safety</strong> - Prevent harm, have safeguards</li>
        <li><strong>Accountability</strong> - Clear responsibility and recourse</li>
        <li><strong>Human oversight</strong> - Humans in the loop for important decisions</li>
      </ul>

      <h3>Regulations and Governance</h3>

      <h4>Emerging Regulations:</h4>
      <ul>
        <li><strong>EU AI Act</strong> - Risk-based regulation of AI systems</li>
        <li><strong>US Executive Order on AI</strong> - Guidelines for federal AI use</li>
        <li><strong>China AI Regulations</strong> - Rules for generative AI</li>
      </ul>

      <h4>Industry Self-Regulation:</h4>
      <ul>
        <li>AI safety research organizations (Anthropic, OpenAI safety teams)</li>
        <li>Ethics boards and review processes</li>
        <li>Voluntary commitments and standards</li>
      </ul>

      <h3>Best Practices for Developers</h3>
      <ul>
        <li>Test for bias across different demographic groups</li>
        <li>Be transparent about AI use and limitations</li>
        <li>Implement robust security and safety measures</li>
        <li>Allow for human review and override</li>
        <li>Document training data and model decisions</li>
        <li>Consider societal impact beyond immediate use case</li>
        <li>Engage with affected communities</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. AI can perpetuate bias present in training data</p>
        <p>2. Privacy, transparency, and accountability are key ethical concerns</p>
        <p>3. AI alignment - ensuring AI does what we intend - is a major challenge</p>
        <p>4. Responsible AI requires fairness, transparency, safety, and human oversight</p>
        <p>5. Both regulation and industry practices are evolving to address these issues</p>
      </div>
    `,
    questions: [
      {
        id: "8-1",
        type: "mcq",
        question: "What is 'bias' in AI systems?",
        options: [
          "A technical error in the code",
          "When AI systems treat different groups unfairly due to patterns in training data",
          "When AI is too slow",
          "A type of neural network architecture"
        ],
        correctAnswer: "When AI systems treat different groups unfairly due to patterns in training data",
        explanation: "AI bias occurs when systems treat different groups unfairly, often because they learned biased patterns from training data. This can result in discrimination against certain demographics."
      },
      {
        id: "8-2",
        type: "mcq",
        question: "What is the AI alignment problem?",
        options: [
          "Making sure AI code is properly formatted",
          "Ensuring AI systems reliably pursue goals that match human values",
          "Aligning multiple AI models together",
          "Physical alignment of hardware"
        ],
        correctAnswer: "Ensuring AI systems reliably pursue goals that match human values",
        explanation: "The alignment problem is the challenge of ensuring AI systems reliably pursue goals that match human intentions and values, rather than finding unexpected or harmful ways to achieve objectives."
      },
      {
        id: "8-3",
        type: "mcq",
        question: "What is 'Constitutional AI'?",
        options: [
          "AI that follows government laws",
          "An approach where AI is trained to follow a set of principles",
          "AI used in government",
          "A legal framework for AI"
        ],
        correctAnswer: "An approach where AI is trained to follow a set of principles",
        explanation: "Constitutional AI is an alignment approach developed by Anthropic where AI is trained to follow a set of principles (a 'constitution') that guide its behavior, helping it refuse harmful requests and be more helpful."
      },
      {
        id: "8-4",
        type: "descriptive",
        question: "What are 3 key ethical concerns in AI development and why do they matter?",
        keywords: ["bias", "privacy", "transparency", "fairness", "accountability", "discrimination", "data", "explainability", "black box"],
        explanation: "Key ethical concerns: 1) Bias - AI can discriminate against groups, causing real-world harm. 2) Privacy - AI requires lots of data and can enable surveillance. 3) Transparency - 'Black box' models make it hard to understand or challenge decisions, especially in high-stakes contexts like healthcare or justice."
      },
      {
        id: "8-5",
        type: "descriptive",
        question: "What practices should developers follow to build more responsible AI systems?",
        keywords: ["bias", "test", "transparent", "security", "human", "oversight", "document", "review", "impact", "community"],
        explanation: "Responsible AI practices include: testing for bias across demographics, being transparent about AI use and limitations, implementing security measures, allowing human review/override, documenting training data and decisions, considering societal impact, and engaging affected communities."
      }
    ]
  },
  {
    id: 9,
    title: "The Future of AI",
    description: "Explore emerging trends and where AI is heading",
    duration: "35 min",
    difficulty: "Beginner",
    content: `
      <h2>Current State of AI</h2>
      <p>Before looking ahead, let's understand where we are:</p>
      <ul>
        <li>LLMs can write, code, reason, and pass professional exams</li>
        <li>Image generation creates photorealistic content</li>
        <li>AI assistants are becoming mainstream</li>
        <li>Autonomous systems are improving rapidly</li>
      </ul>

      <h3>Emerging Trends</h3>

      <h4>1. Multimodal AI</h4>
      <p>AI that works across multiple types of data:</p>
      <ul>
        <li><strong>Text + Image</strong> - GPT-4V, Claude with vision</li>
        <li><strong>Text + Audio</strong> - Speech understanding and generation</li>
        <li><strong>Text + Video</strong> - Video understanding and generation (Sora)</li>
        <li><strong>Any-to-any</strong> - Future systems that handle all modalities</li>
      </ul>

      <h4>2. AI Agents and Automation</h4>
      <p>From chatbots to agents that take action:</p>
      <ul>
        <li>Computer use (browsing, clicking, typing)</li>
        <li>Code writing and execution</li>
        <li>Multi-step task completion</li>
        <li>Autonomous research and planning</li>
      </ul>

      <h4>3. Smaller, More Efficient Models</h4>
      <p>Not just bigger is better:</p>
      <ul>
        <li><strong>Distillation</strong> - Transferring knowledge to smaller models</li>
        <li><strong>Quantization</strong> - Reducing precision for efficiency</li>
        <li><strong>On-device AI</strong> - Running on phones and edge devices</li>
        <li><strong>Specialized models</strong> - Task-specific efficient architectures</li>
      </ul>

      <h4>4. Improved Reasoning</h4>
      <p>Moving beyond pattern matching:</p>
      <ul>
        <li>Chain-of-thought and step-by-step reasoning</li>
        <li>Self-correction and reflection</li>
        <li>Longer context and better memory</li>
        <li>More reliable logical inference</li>
      </ul>

      <h4>5. AI in Science</h4>
      <p>Accelerating scientific discovery:</p>
      <ul>
        <li><strong>AlphaFold</strong> - Protein structure prediction</li>
        <li><strong>Drug discovery</strong> - Finding new medications faster</li>
        <li><strong>Materials science</strong> - Discovering new materials</li>
        <li><strong>Climate modeling</strong> - Better predictions and solutions</li>
      </ul>

      <h4>6. Personalized AI</h4>
      <p>AI tailored to individuals:</p>
      <ul>
        <li>Personal assistants that know your preferences</li>
        <li>Customized learning experiences</li>
        <li>Health AI with personal medical history</li>
        <li>AI that adapts to your communication style</li>
      </ul>

      <h3>Near-term Predictions (1-3 years)</h3>
      <ul>
        <li>AI assistants become standard in most software</li>
        <li>Code generation becomes routine in development</li>
        <li>Multimodal understanding becomes standard</li>
        <li>AI tutoring transforms education</li>
        <li>Autonomous agents handle routine tasks</li>
        <li>Real-time translation becomes seamless</li>
      </ul>

      <h3>Medium-term Possibilities (3-10 years)</h3>
      <ul>
        <li>Highly capable AI assistants for complex tasks</li>
        <li>Significant automation of knowledge work</li>
        <li>Major breakthroughs in scientific research</li>
        <li>Autonomous vehicles become widespread</li>
        <li>AI-human collaboration becomes the norm</li>
        <li>Potential for AGI (Artificial General Intelligence)</li>
      </ul>

      <h3>Challenges Ahead</h3>

      <h4>Technical Challenges:</h4>
      <ul>
        <li>Reliable reasoning and reduced hallucinations</li>
        <li>Long-term memory and learning</li>
        <li>Energy efficiency and environmental impact</li>
        <li>Robust security against attacks</li>
      </ul>

      <h4>Societal Challenges:</h4>
      <ul>
        <li>Job displacement and economic disruption</li>
        <li>Misinformation and deepfakes</li>
        <li>Privacy in an AI-powered world</li>
        <li>Ensuring equitable access to AI benefits</li>
        <li>Maintaining human agency and purpose</li>
      </ul>

      <h3>Preparing for an AI Future</h3>

      <h4>For Individuals:</h4>
      <ul>
        <li><strong>Learn to use AI tools</strong> - Become AI-literate</li>
        <li><strong>Develop complementary skills</strong> - Creativity, empathy, judgment</li>
        <li><strong>Stay adaptable</strong> - Embrace continuous learning</li>
        <li><strong>Understand limitations</strong> - Know when not to trust AI</li>
      </ul>

      <h4>For Organizations:</h4>
      <ul>
        <li>Invest in AI capabilities and infrastructure</li>
        <li>Reskill workforce for AI collaboration</li>
        <li>Develop AI governance and ethics frameworks</li>
        <li>Plan for disruption in your industry</li>
      </ul>

      <h4>For Society:</h4>
      <ul>
        <li>Update education for an AI world</li>
        <li>Develop appropriate regulations</li>
        <li>Address job displacement proactively</li>
        <li>Ensure AI benefits are widely shared</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. Multimodal AI, agents, and efficient models are key trends</p>
        <p>2. AI will transform work, science, and daily life</p>
        <p>3. Both technical and societal challenges must be addressed</p>
        <p>4. AI literacy and adaptability are essential skills</p>
        <p>5. The future depends on choices we make today</p>
      </div>
    `,
    questions: [
      {
        id: "9-1",
        type: "mcq",
        question: "What is 'multimodal AI'?",
        options: [
          "AI that runs on multiple computers",
          "AI that can process multiple types of data (text, images, audio, etc.)",
          "AI with multiple training phases",
          "AI that speaks multiple languages"
        ],
        correctAnswer: "AI that can process multiple types of data (text, images, audio, etc.)",
        explanation: "Multimodal AI can understand and generate multiple types of data - text, images, audio, and video. Examples include GPT-4V (text + vision) and models like Gemini that handle multiple modalities."
      },
      {
        id: "9-2",
        type: "mcq",
        question: "What are AI agents?",
        options: [
          "Human employees who work with AI",
          "AI systems that can take actions and complete tasks autonomously",
          "AI models that are very large",
          "Chatbots that answer questions"
        ],
        correctAnswer: "AI systems that can take actions and complete tasks autonomously",
        explanation: "AI agents go beyond chatbots - they can take actions like browsing the web, writing and executing code, and completing multi-step tasks autonomously, rather than just generating text responses."
      },
      {
        id: "9-3",
        type: "mcq",
        question: "What is a key near-term prediction for AI (1-3 years)?",
        options: [
          "AI will achieve consciousness",
          "AI assistants will become standard in most software",
          "AI will replace all human jobs",
          "AI will solve climate change"
        ],
        correctAnswer: "AI assistants will become standard in most software",
        explanation: "AI assistants are already being integrated into many software products and this trend is expected to accelerate. Within 1-3 years, AI assistance will likely be a standard feature in most software applications."
      },
      {
        id: "9-4",
        type: "descriptive",
        question: "What skills should individuals develop to prepare for an AI-powered future?",
        keywords: ["learn", "AI", "tool", "adapt", "creative", "empathy", "judgment", "continuous", "learning", "limitation"],
        explanation: "To prepare for an AI future, individuals should: 1) Learn to use AI tools effectively, 2) Develop complementary skills AI lacks (creativity, empathy, judgment), 3) Stay adaptable with continuous learning, 4) Understand AI limitations and when not to trust it."
      },
      {
        id: "9-5",
        type: "descriptive",
        question: "What are major societal challenges that AI presents?",
        keywords: ["job", "displacement", "misinformation", "deepfake", "privacy", "access", "equitable", "economic", "human", "agency"],
        explanation: "Major societal challenges include: job displacement and economic disruption, misinformation and deepfakes, privacy concerns, ensuring equitable access to AI benefits, and maintaining human agency and purpose in an AI-powered world."
      }
    ]
  },
  {
    id: 10,
    title: "Hands-on: Building Your First AI App",
    description: "Put your knowledge into practice with a guided project",
    duration: "90 min",
    difficulty: "Advanced",
    content: `
      <h2>Project: Build an AI-Powered Q&A System</h2>
      <p>In this hands-on lesson, we'll build a complete AI application that can answer questions about any documents you provide.</p>

      <h3>What We'll Build</h3>
      <p>A RAG (Retrieval-Augmented Generation) system with:</p>
      <ul>
        <li>Document upload and processing</li>
        <li>Vector storage for semantic search</li>
        <li>AI-powered question answering</li>
        <li>Web interface for interaction</li>
      </ul>

      <h3>Technology Stack</h3>
      <ul>
        <li><strong>Next.js</strong> - React framework for the web app</li>
        <li><strong>Claude API</strong> - For generating answers</li>
        <li><strong>OpenAI Embeddings</strong> - For converting text to vectors</li>
        <li><strong>Supabase pgvector</strong> - Vector database</li>
      </ul>

      <h3>Step 1: Project Setup</h3>
      <div class="code-block">
npx create-next-app@latest my-ai-app --typescript --tailwind
cd my-ai-app
npm install @anthropic-ai/sdk openai @supabase/supabase-js
      </div>

      <h3>Step 2: Database Setup</h3>
      <p>Create a Supabase project and run this SQL:</p>
      <div class="code-block">
-- Enable the pgvector extension
create extension if not exists vector;

-- Create table for document chunks
create table document_chunks (
  id uuid default gen_random_uuid() primary key,
  content text not null,
  embedding vector(1536),
  metadata jsonb,
  created_at timestamp with time zone default now()
);

-- Create function for similarity search
create function match_documents(
  query_embedding vector(1536),
  match_count int default 5
) returns table (
  id uuid,
  content text,
  similarity float
)
language plpgsql as $$
begin
  return query
  select
    document_chunks.id,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
      </div>

      <h3>Step 3: Environment Variables</h3>
      <div class="code-block">
# .env.local
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
      </div>

      <h3>Step 4: Create Embedding Function</h3>
      <div class="code-block">
// lib/embeddings.ts
import OpenAI from 'openai';

const openai = new OpenAI();

export async function createEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}
      </div>

      <h3>Step 5: Document Processing</h3>
      <div class="code-block">
// lib/documents.ts
import { createClient } from '@supabase/supabase-js';
import { createEmbedding } from './embeddings';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_KEY!
);

export async function processDocument(content: string, chunkSize = 500) {
  // Split into chunks
  const chunks = [];
  for (let i = 0; i < content.length; i += chunkSize) {
    chunks.push(content.slice(i, i + chunkSize));
  }

  // Create embeddings and store
  for (const chunk of chunks) {
    const embedding = await createEmbedding(chunk);
    await supabase.from('document_chunks').insert({
      content: chunk,
      embedding: embedding,
    });
  }
}

export async function searchDocuments(query: string, count = 5) {
  const queryEmbedding = await createEmbedding(query);

  const { data, error } = await supabase.rpc('match_documents', {
    query_embedding: queryEmbedding,
    match_count: count,
  });

  if (error) throw error;
  return data;
}
      </div>

      <h3>Step 6: Q&A API Route</h3>
      <div class="code-block">
// app/api/ask/route.ts
import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { searchDocuments } from '@/lib/documents';

const anthropic = new Anthropic();

export async function POST(req: Request) {
  const { question } = await req.json();

  // Find relevant documents
  const relevantDocs = await searchDocuments(question);
  const context = relevantDocs.map(d => d.content).join('\\n\\n');

  // Generate answer with Claude
  const message = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 1024,
    messages: [{
      role: 'user',
      content: \`Based on the following context, answer the question.

Context:
\${context}

Question: \${question}

Answer based only on the provided context. If the answer isn't in the context, say so.\`
    }]
  });

  return NextResponse.json({
    answer: message.content[0].text,
    sources: relevantDocs
  });
}
      </div>

      <h3>Step 7: Frontend Interface</h3>
      <div class="code-block">
// app/page.tsx
'use client';
import { useState } from 'react';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleAsk() {
    setLoading(true);
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  }

  return (
    &lt;main className="max-w-2xl mx-auto p-8"&gt;
      &lt;h1 className="text-3xl font-bold mb-8"&gt;AI Document Q&A&lt;/h1&gt;

      &lt;div className="space-y-4"&gt;
        &lt;textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="w-full p-4 border rounded-lg"
          rows={3}
        /&gt;

        &lt;button
          onClick={handleAsk}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg"
        &gt;
          {loading ? 'Thinking...' : 'Ask'}
        &lt;/button&gt;

        {answer && (
          &lt;div className="p-4 bg-gray-100 rounded-lg"&gt;
            &lt;h2 className="font-bold mb-2"&gt;Answer:&lt;/h2&gt;
            &lt;p&gt;{answer}&lt;/p&gt;
          &lt;/div&gt;
        )}
      &lt;/div&gt;
    &lt;/main&gt;
  );
}
      </div>

      <h3>Key Concepts Applied</h3>
      <ul>
        <li><strong>RAG</strong> - Retrieval-Augmented Generation pattern</li>
        <li><strong>Embeddings</strong> - Converting text to vectors for similarity search</li>
        <li><strong>Vector Database</strong> - Storing and searching embeddings</li>
        <li><strong>Prompt Engineering</strong> - Crafting effective prompts with context</li>
        <li><strong>API Integration</strong> - Using LLM APIs in applications</li>
      </ul>

      <h3>Extending the Project</h3>
      <p>Ideas for improvement:</p>
      <ul>
        <li>Add file upload for PDFs, Word docs</li>
        <li>Implement conversation history</li>
        <li>Add source citations to answers</li>
        <li>Create admin interface for managing documents</li>
        <li>Add user authentication</li>
        <li>Implement streaming responses</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. RAG combines retrieval and generation for knowledge-grounded answers</p>
        <p>2. Embeddings enable semantic similarity search</p>
        <p>3. Chunking documents properly is crucial for good retrieval</p>
        <p>4. Prompt engineering determines answer quality</p>
        <p>5. This pattern can be adapted for many use cases</p>
      </div>
    `,
    questions: [
      {
        id: "10-1",
        type: "mcq",
        question: "Why do we split documents into chunks before storing them?",
        options: [
          "To make the database smaller",
          "Because LLMs have context limits and smaller chunks allow for more precise retrieval",
          "To make embedding faster",
          "It's not actually necessary"
        ],
        correctAnswer: "Because LLMs have context limits and smaller chunks allow for more precise retrieval",
        explanation: "Documents are chunked because: 1) LLMs have context limits, so we need to select the most relevant portions, 2) Smaller chunks allow for more precise retrieval - a relevant paragraph is better than a slightly relevant page."
      },
      {
        id: "10-2",
        type: "mcq",
        question: "What is the role of embeddings in a RAG system?",
        options: [
          "To compress the text",
          "To encrypt the data",
          "To convert text into vectors for semantic similarity search",
          "To format the output"
        ],
        correctAnswer: "To convert text into vectors for semantic similarity search",
        explanation: "Embeddings convert text into numerical vectors where semantically similar texts have similar vectors. This enables finding relevant documents based on meaning, not just keyword matching."
      },
      {
        id: "10-3",
        type: "descriptive",
        question: "Describe the flow of a RAG system from user question to answer.",
        keywords: ["question", "embedding", "vector", "search", "retrieve", "context", "prompt", "LLM", "generate", "answer"],
        explanation: "RAG flow: 1) User asks question, 2) Question converted to embedding, 3) Vector similarity search finds relevant document chunks, 4) Retrieved chunks added to prompt as context, 5) LLM generates answer based on context, 6) Answer returned to user."
      },
      {
        id: "10-4",
        type: "descriptive",
        question: "What are 3 ways you could extend or improve this basic RAG application?",
        keywords: ["file", "upload", "PDF", "history", "conversation", "citation", "source", "authentication", "streaming", "memory"],
        explanation: "Possible extensions: 1) Add file upload for various formats (PDF, Word), 2) Implement conversation history/memory, 3) Add source citations showing where answers came from, 4) Create document management interface, 5) Add user authentication, 6) Implement streaming responses."
      }
    ]
  }
];

export const getLessonById = (id: number): Lesson | undefined => {
  return lessons.find(lesson => lesson.id === id);
};
