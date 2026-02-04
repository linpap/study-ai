import { Lesson } from './lessons';

// These lessons should be inserted between existing lessons to provide proper foundation
export const foundationalLessons: Lesson[] = [
  {
    id: 3,
    title: "Math Foundations: Linear Algebra Essentials",
    description: "Master the mathematical building blocks of AI - vectors, matrices, and transformations",
    duration: "50 min",
    difficulty: "Beginner",
    content: `
      <h2>Why Linear Algebra for AI?</h2>
      <p>Linear algebra is the backbone of machine learning. Every image, text, and data point is represented as numbers in vectors and matrices. Understanding these concepts is crucial for understanding how AI actually works.</p>

      <h3>Vectors: The Basic Building Block</h3>
      <p>A <strong>vector</strong> is simply a list of numbers. In AI, vectors represent data points.</p>

      <div class="code-block">
        <pre>
# A 3-dimensional vector
v = [1, 2, 3]

# In AI, vectors represent features
# Example: A person represented as [age, height_cm, weight_kg]
person = [25, 175, 70]

# An image pixel: [red, green, blue]
pixel = [255, 128, 0]  # Orange color
        </pre>
      </div>

      <h4>Vector Operations</h4>
      <p><strong>Addition:</strong> Add corresponding elements</p>
      <div class="code-block">
        <pre>
[1, 2, 3] + [4, 5, 6] = [5, 7, 9]
        </pre>
      </div>

      <p><strong>Scalar Multiplication:</strong> Multiply each element by a number</p>
      <div class="code-block">
        <pre>
2 × [1, 2, 3] = [2, 4, 6]
        </pre>
      </div>

      <p><strong>Dot Product:</strong> Multiply corresponding elements and sum them</p>
      <div class="code-block">
        <pre>
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

# The dot product measures similarity between vectors!
# Higher dot product = more similar direction
        </pre>
      </div>

      <div class="highlight">
        <p><strong>Why Dot Products Matter in AI:</strong></p>
        <ul>
          <li>Attention mechanisms in Transformers use dot products to measure relevance</li>
          <li>Neural networks compute dot products between weights and inputs</li>
          <li>Similarity search (like finding similar documents) uses dot products</li>
        </ul>
      </div>

      <h3>Matrices: Organized Data</h3>
      <p>A <strong>matrix</strong> is a 2D array of numbers - like a spreadsheet. Matrices store datasets and model parameters.</p>

      <div class="code-block">
        <pre>
# A 3×2 matrix (3 rows, 2 columns)
M = [
  [1, 2],
  [3, 4],
  [5, 6]
]

# A dataset of 3 people with 2 features each (age, income)
dataset = [
  [25, 50000],
  [30, 75000],
  [35, 90000]
]
        </pre>
      </div>

      <h4>Matrix Multiplication</h4>
      <p>Matrix multiplication is how neural networks process data. Each layer multiplies inputs by weights.</p>

      <div class="code-block">
        <pre>
# Input vector: [x1, x2]
# Weight matrix: 2 inputs → 3 outputs
W = [
  [w11, w12],  # weights for output 1
  [w21, w22],  # weights for output 2
  [w31, w32]   # weights for output 3
]

# Output = W × Input
# Each output is a dot product of weights with input
        </pre>
      </div>

      <h3>Practical Example: Image as Matrix</h3>
      <p>A grayscale image is a matrix where each cell is a pixel brightness (0-255):</p>
      <div class="code-block">
        <pre>
# 4×4 grayscale image
image = [
  [0,   0,   255, 255],   # Top row: black black white white
  [0,   0,   255, 255],
  [255, 255, 0,   0  ],
  [255, 255, 0,   0  ]    # Bottom row: white white black black
]

# Color image: 3 matrices (Red, Green, Blue channels)
# Shape: height × width × 3
        </pre>
      </div>

      <h3>Norms: Measuring Size</h3>
      <p>The <strong>norm</strong> measures the "length" or "magnitude" of a vector.</p>

      <p><strong>L2 Norm (Euclidean):</strong> Most common - actual geometric length</p>
      <div class="code-block">
        <pre>
||[3, 4]|| = √(3² + 4²) = √(9 + 16) = √25 = 5
        </pre>
      </div>

      <p><strong>Used in AI for:</strong></p>
      <ul>
        <li>Regularization (preventing overfitting)</li>
        <li>Normalizing vectors (making length = 1)</li>
        <li>Measuring distances between data points</li>
      </ul>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Vectors</strong> represent data points and features</p>
        <p>2. <strong>Dot products</strong> measure similarity - core to attention mechanisms</p>
        <p>3. <strong>Matrices</strong> store datasets and neural network weights</p>
        <p>4. <strong>Matrix multiplication</strong> is how neural networks process data</p>
        <p>5. <strong>Norms</strong> measure magnitude - used in regularization</p>
      </div>
    `,
    questions: [
      {
        id: "3-1",
        type: "mcq",
        question: "What does the dot product of two vectors measure?",
        options: ["Their sum", "Their similarity/alignment", "Their difference", "Their product"],
        correctAnswer: "Their similarity/alignment",
        explanation: "The dot product measures how similar or aligned two vectors are. A higher dot product means the vectors point in similar directions. This is why attention mechanisms use dot products to measure relevance between query and key vectors."
      },
      {
        id: "3-2",
        type: "mcq",
        question: "In a neural network, what happens when input data is multiplied by weight matrices?",
        options: ["Data is deleted", "Data is transformed to new representations", "Data stays the same", "Data is compressed only"],
        correctAnswer: "Data is transformed to new representations",
        explanation: "Matrix multiplication transforms input data into new representations. Each layer learns different features by adjusting its weight matrix during training."
      },
      {
        id: "3-3",
        type: "descriptive",
        question: "Explain why understanding linear algebra is important for AI, giving at least 2 specific examples.",
        keywords: ["vector", "matrix", "dot product", "similarity", "attention", "neural network", "weights", "transformation", "data", "representation"],
        explanation: "Linear algebra is essential because: (1) Data is represented as vectors/matrices - images are matrices, text becomes vectors. (2) Neural networks use matrix multiplication to transform data through layers. (3) Attention mechanisms use dot products to measure similarity between elements. (4) Understanding these concepts helps debug and improve models."
      },
      {
        id: "3-4",
        type: "mcq",
        question: "How is a color image typically represented mathematically?",
        options: ["A single vector", "A 2D matrix", "A 3D tensor (height × width × 3)", "A 1D array"],
        correctAnswer: "A 3D tensor (height × width × 3)",
        explanation: "A color image is represented as a 3D tensor with dimensions height × width × 3, where the 3 represents the Red, Green, and Blue color channels."
      }
    ]
  },
  {
    id: 4,
    title: "Math Foundations: Calculus for Machine Learning",
    description: "Understand gradients and optimization - how neural networks learn",
    duration: "45 min",
    difficulty: "Beginner",
    content: `
      <h2>Why Calculus for AI?</h2>
      <p>Calculus answers one crucial question: <strong>How do we improve?</strong> In machine learning, we need to minimize errors, and calculus tells us which direction to adjust our parameters.</p>

      <h3>The Core Idea: Derivatives</h3>
      <p>A <strong>derivative</strong> measures how much a function's output changes when you slightly change its input. It's the "slope" or "rate of change."</p>

      <div class="code-block">
        <pre>
# If f(x) = x², the derivative f'(x) = 2x

# At x = 3:
# f(3) = 9
# f'(3) = 6  ← The slope at this point

# This means: if you increase x by a tiny amount,
# the output increases by about 6 times that amount
        </pre>
      </div>

      <h3>Gradients: Multi-dimensional Derivatives</h3>
      <p>When a function has multiple inputs, the <strong>gradient</strong> is a vector of all partial derivatives - one for each input.</p>

      <div class="code-block">
        <pre>
# Loss function with 2 weights: L(w1, w2)
# Gradient: [∂L/∂w1, ∂L/∂w2]

# This tells us:
# - How much does loss change if we adjust w1?
# - How much does loss change if we adjust w2?
        </pre>
      </div>

      <div class="highlight">
        <p><strong>Key Insight:</strong> The gradient points in the direction of steepest INCREASE. To minimize loss, we go in the OPPOSITE direction (negative gradient).</p>
      </div>

      <h3>Gradient Descent: How Neural Networks Learn</h3>
      <p>Gradient descent is the algorithm that trains neural networks:</p>

      <div class="code-block">
        <pre>
# The learning algorithm:
repeat until converged:
    1. Calculate predictions with current weights
    2. Calculate loss (how wrong we are)
    3. Calculate gradient (which direction to adjust)
    4. Update weights: w_new = w_old - learning_rate × gradient

# Example with learning_rate = 0.1:
# If gradient = [2, -3]
# w1_new = w1_old - 0.1 × 2 = w1_old - 0.2
# w2_new = w2_old - 0.1 × (-3) = w2_old + 0.3
        </pre>
      </div>

      <h4>Visualizing Gradient Descent</h4>
      <p>Imagine standing on a hilly landscape (the loss surface) and wanting to reach the lowest point (minimum loss):</p>
      <ul>
        <li>The gradient tells you which way is "uphill"</li>
        <li>You walk in the opposite direction (downhill)</li>
        <li>Learning rate controls your step size</li>
        <li>Too big → overshoot the minimum</li>
        <li>Too small → takes forever to converge</li>
      </ul>

      <h3>The Chain Rule: Backpropagation's Foundation</h3>
      <p>The <strong>chain rule</strong> lets us calculate gradients through multiple layers:</p>

      <div class="code-block">
        <pre>
# If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x)

# Neural Network Example:
# Input → Layer1 → Layer2 → Output → Loss

# To find how input affects loss:
# dLoss/dInput = dLoss/dOutput × dOutput/dLayer2 × dLayer2/dLayer1 × dLayer1/dInput

# This "chain" of derivatives is backpropagation!
        </pre>
      </div>

      <h3>Common Loss Functions</h3>
      <p>Loss functions measure prediction errors. Their derivatives tell us how to improve:</p>

      <h4>Mean Squared Error (MSE)</h4>
      <div class="code-block">
        <pre>
# For regression (predicting numbers)
MSE = (1/n) × Σ(predicted - actual)²

# Derivative w.r.t. prediction:
d(MSE)/d(predicted) = (2/n) × (predicted - actual)
        </pre>
      </div>

      <h4>Cross-Entropy Loss</h4>
      <div class="code-block">
        <pre>
# For classification (predicting categories)
# Penalizes confident wrong predictions heavily
CrossEntropy = -Σ actual × log(predicted)
        </pre>
      </div>

      <h3>Learning Rate: The Critical Hyperparameter</h3>
      <div class="code-block">
        <pre>
# Too high learning rate (e.g., 1.0):
# - Overshoots the minimum
# - Loss oscillates or explodes
# - Training fails

# Too low learning rate (e.g., 0.00001):
# - Takes forever to converge
# - Might get stuck in local minima
# - Wastes compute

# Good starting points:
# - 0.001 for Adam optimizer
# - 0.01 for SGD
# - Use learning rate schedulers to decrease over time
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Derivatives</strong> measure rate of change - how sensitive output is to input</p>
        <p>2. <strong>Gradients</strong> point toward steepest increase - we go opposite to minimize</p>
        <p>3. <strong>Gradient descent</strong> iteratively updates weights to reduce loss</p>
        <p>4. <strong>Chain rule</strong> enables backpropagation through multiple layers</p>
        <p>5. <strong>Learning rate</strong> controls step size - critical to get right</p>
      </div>
    `,
    questions: [
      {
        id: "4-1",
        type: "mcq",
        question: "What does the gradient of a loss function tell us?",
        options: ["The current loss value", "The direction to adjust weights to INCREASE loss", "The direction to adjust weights to DECREASE loss", "The number of training examples"],
        correctAnswer: "The direction to adjust weights to INCREASE loss",
        explanation: "The gradient points in the direction of steepest INCREASE. That's why in gradient descent, we move in the OPPOSITE direction (subtract the gradient) to minimize loss."
      },
      {
        id: "4-2",
        type: "mcq",
        question: "What happens if the learning rate is too high?",
        options: ["Training is too slow", "Model overfits", "Training overshoots the minimum and may fail to converge", "Model underfits"],
        correctAnswer: "Training overshoots the minimum and may fail to converge",
        explanation: "A learning rate that's too high causes the weight updates to be too large, overshooting the optimal values. This can cause the loss to oscillate or even explode, preventing convergence."
      },
      {
        id: "4-3",
        type: "descriptive",
        question: "Explain the gradient descent algorithm in simple terms. What are the key steps?",
        keywords: ["prediction", "loss", "gradient", "derivative", "update", "weights", "learning rate", "minimize", "direction", "iterate"],
        explanation: "Gradient descent: (1) Make predictions with current weights, (2) Calculate loss (error), (3) Calculate gradient (direction of steepest increase), (4) Update weights in opposite direction: w = w - learning_rate × gradient, (5) Repeat until loss is minimized."
      },
      {
        id: "4-4",
        type: "mcq",
        question: "What mathematical concept allows backpropagation to compute gradients through multiple neural network layers?",
        options: ["Integration", "The chain rule", "Matrix inversion", "Fourier transform"],
        correctAnswer: "The chain rule",
        explanation: "The chain rule allows us to compute how changes in early layers affect the final loss by multiplying derivatives through each layer: dLoss/dInput = dLoss/dOutput × dOutput/dLayer2 × ..."
      }
    ]
  },
  {
    id: 5,
    title: "Math Foundations: Probability & Statistics",
    description: "Learn the probabilistic foundations that power AI predictions and uncertainty",
    duration: "45 min",
    difficulty: "Beginner",
    content: `
      <h2>Why Probability in AI?</h2>
      <p>AI deals with uncertainty. Models don't just predict - they estimate probabilities. Understanding probability helps you:</p>
      <ul>
        <li>Interpret model outputs correctly</li>
        <li>Handle noisy, incomplete data</li>
        <li>Make decisions under uncertainty</li>
        <li>Understand concepts like overfitting and generalization</li>
      </ul>

      <h3>Basic Probability</h3>
      <p>Probability measures how likely an event is, from 0 (impossible) to 1 (certain).</p>

      <div class="code-block">
        <pre>
# Probability basics
P(rain tomorrow) = 0.3  # 30% chance of rain
P(sunny) = 0.7          # 70% chance of sun

# Rule 1: Probabilities sum to 1
# P(all possible outcomes) = 1

# Rule 2: Complement
# P(not A) = 1 - P(A)
# P(no rain) = 1 - 0.3 = 0.7
        </pre>
      </div>

      <h3>Conditional Probability: The Heart of ML</h3>
      <p><strong>P(A|B)</strong> means "probability of A given that B happened"</p>

      <div class="code-block">
        <pre>
# Email spam classification
P(spam | contains "free money") = 0.95
# "Given that an email contains 'free money',
#  there's 95% chance it's spam"

# This is what classifiers learn!
# They estimate P(class | features)
        </pre>
      </div>

      <h3>Bayes' Theorem: Updating Beliefs</h3>
      <p>Bayes' theorem tells us how to update our beliefs with new evidence:</p>

      <div class="code-block">
        <pre>
P(A|B) = P(B|A) × P(A) / P(B)

# In ML terms:
P(class|data) = P(data|class) × P(class) / P(data)

# Example: Disease diagnosis
# P(disease | positive test) =
#   P(positive test | disease) × P(disease) / P(positive test)

# Even if test is 99% accurate, if disease is rare (1%),
# a positive test might only mean 50% chance of disease!
        </pre>
      </div>

      <div class="highlight">
        <p><strong>Naive Bayes Classifier</strong> uses Bayes' theorem directly! It assumes features are independent and calculates P(class|features) for each class.</p>
      </div>

      <h3>Probability Distributions</h3>
      <p>Distributions describe the possible values and their probabilities:</p>

      <h4>Normal (Gaussian) Distribution</h4>
      <div class="code-block">
        <pre>
# Bell curve - most common in nature
# Defined by mean (μ) and standard deviation (σ)

# 68% of data within 1σ of mean
# 95% of data within 2σ of mean
# 99.7% of data within 3σ of mean

# Used in:
# - Weight initialization in neural networks
# - Noise in generative models
# - Uncertainty estimation
        </pre>
      </div>

      <h4>Softmax: Turning Scores into Probabilities</h4>
      <div class="code-block">
        <pre>
# Neural network outputs raw scores (logits)
logits = [2.0, 1.0, 0.5]

# Softmax converts to probabilities
# softmax(x_i) = exp(x_i) / Σexp(x_j)

probabilities = [0.59, 0.24, 0.17]  # Sum = 1.0

# Now we have a proper probability distribution!
        </pre>
      </div>

      <h3>Key Statistical Concepts</h3>

      <h4>Mean (Average)</h4>
      <div class="code-block">
        <pre>
data = [1, 2, 3, 4, 5]
mean = (1+2+3+4+5) / 5 = 3
        </pre>
      </div>

      <h4>Variance & Standard Deviation</h4>
      <div class="code-block">
        <pre>
# Variance: average squared distance from mean
variance = Σ(x - mean)² / n

# Standard deviation: square root of variance
# Measures "spread" of data

# High variance = data is spread out
# Low variance = data is clustered around mean
        </pre>
      </div>

      <h4>Correlation</h4>
      <div class="code-block">
        <pre>
# Measures linear relationship between variables
# Range: -1 to +1

correlation = +1   # Perfect positive (both increase together)
correlation = -1   # Perfect negative (one up, other down)
correlation = 0    # No linear relationship

# Important: Correlation ≠ Causation!
        </pre>
      </div>

      <h3>Probability in Neural Network Outputs</h3>
      <div class="code-block">
        <pre>
# Classification output (after softmax):
{
  "cat": 0.75,      # 75% confident it's a cat
  "dog": 0.20,      # 20% confident it's a dog
  "bird": 0.05      # 5% confident it's a bird
}

# Regression with uncertainty:
prediction = {
  "mean": 150000,           # Predicted house price
  "std": 20000              # Uncertainty (±$20k)
}
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Probability</strong> quantifies uncertainty (0 to 1)</p>
        <p>2. <strong>Conditional probability</strong> P(A|B) is what classifiers learn</p>
        <p>3. <strong>Bayes' theorem</strong> updates beliefs with evidence</p>
        <p>4. <strong>Softmax</strong> converts neural network outputs to probabilities</p>
        <p>5. <strong>Variance/std</strong> measure spread and uncertainty</p>
      </div>
    `,
    questions: [
      {
        id: "5-1",
        type: "mcq",
        question: "What does P(spam|email_contains_prize) represent?",
        options: ["Probability that any email is spam", "Probability that spam emails contain 'prize'", "Probability an email is spam given it contains 'prize'", "Probability of finding 'prize' in any email"],
        correctAnswer: "Probability an email is spam given it contains 'prize'",
        explanation: "P(A|B) is conditional probability - the probability of A given that B is true. P(spam|contains_prize) is the probability the email is spam, given that it contains the word 'prize'."
      },
      {
        id: "5-2",
        type: "mcq",
        question: "What does the softmax function do?",
        options: ["Makes all values positive", "Converts raw scores into a probability distribution that sums to 1", "Reduces dimensionality", "Normalizes to range [0,1] independently"],
        correctAnswer: "Converts raw scores into a probability distribution that sums to 1",
        explanation: "Softmax takes raw neural network outputs (logits) and converts them to probabilities. The output values are all positive and sum to 1, making them a valid probability distribution."
      },
      {
        id: "5-3",
        type: "descriptive",
        question: "Explain why understanding probability is important for interpreting AI model outputs.",
        keywords: ["uncertainty", "confidence", "probability", "softmax", "classification", "prediction", "distribution", "belief", "decision"],
        explanation: "Probability is crucial because: (1) Model outputs are probabilities, not certainties - a 70% prediction isn't guaranteed. (2) We can make better decisions by considering confidence levels. (3) We can combine predictions with prior knowledge using Bayes' theorem. (4) Understanding uncertainty helps avoid overconfident mistakes."
      },
      {
        id: "5-4",
        type: "mcq",
        question: "In the normal distribution, approximately what percentage of data falls within 2 standard deviations of the mean?",
        options: ["68%", "95%", "99.7%", "50%"],
        correctAnswer: "95%",
        explanation: "The 68-95-99.7 rule: 68% within 1σ, 95% within 2σ, and 99.7% within 3σ of the mean. This is important for understanding model uncertainty and outlier detection."
      }
    ]
  },
  {
    id: 6,
    title: "Classical ML: Decision Trees & Random Forests",
    description: "Learn interpretable algorithms that form the foundation of modern ML",
    duration: "55 min",
    difficulty: "Intermediate",
    content: `
      <h2>Why Learn Classical ML?</h2>
      <p>Before deep learning, classical algorithms solved most ML problems - and they still do for many use cases:</p>
      <ul>
        <li>Often work better on small datasets</li>
        <li>More interpretable - you can explain decisions</li>
        <li>Faster to train and deploy</li>
        <li>Great baselines before trying complex models</li>
      </ul>

      <h3>Decision Trees: If-Then Rules</h3>
      <p>A decision tree makes predictions by learning a series of if-then rules from data.</p>

      <div class="code-block">
        <pre>
# Example: Should I play tennis today?
#
#         [Outlook?]
#        /    |     \\
#    Sunny  Overcast  Rainy
#      |       |        |
#  [Humidity?] Yes   [Windy?]
#   /     \\           /    \\
# High   Normal    True   False
#  |       |        |       |
#  No     Yes      No      Yes

# The tree learned these rules from training data!
        </pre>
      </div>

      <h4>How Trees Learn: Finding the Best Splits</h4>
      <p>At each node, the algorithm finds the feature and threshold that best separates the classes:</p>

      <div class="code-block">
        <pre>
# Goal: Maximize "purity" after split
# Metrics used:

# 1. Gini Impurity (used in CART)
# Gini = 1 - Σ(p_i)²
# Perfect purity: Gini = 0 (all same class)
# Maximum impurity: Gini = 0.5 (50/50 split)

# 2. Information Gain / Entropy
# Entropy = -Σ p_i × log(p_i)
# Lower entropy = purer node

# Example split evaluation:
# Before: 50 cats, 50 dogs (Gini = 0.5)
# After split on "weight > 20kg":
#   Left:  5 cats, 45 dogs (Gini = 0.18)
#   Right: 45 cats, 5 dogs (Gini = 0.18)
# Good split! Both children are purer.
        </pre>
      </div>

      <h4>Tree Hyperparameters</h4>
      <div class="code-block">
        <pre>
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,          # Limit tree depth (prevent overfitting)
    min_samples_split=10, # Min samples to split a node
    min_samples_leaf=5,   # Min samples in leaf nodes
    max_features='sqrt'   # Features to consider per split
)
        </pre>
      </div>

      <h3>The Overfitting Problem</h3>
      <p>Deep trees memorize training data but fail on new data:</p>

      <div class="code-block">
        <pre>
# Overfit tree (no limits):
# - Perfectly classifies training data
# - Creates rules like "if user_id == 12345 then spam"
# - Fails on new users

# Underfit tree (too shallow):
# - Too simple to capture patterns
# - High error on both training and test data

# Solution: Limit depth OR use Random Forests
        </pre>
      </div>

      <h3>Random Forests: Wisdom of the Crowd</h3>
      <p>A Random Forest combines many decision trees to make better predictions:</p>

      <div class="code-block">
        <pre>
# Random Forest = Many Trees + Voting

# How it works:
# 1. Create N different trees (e.g., 100 trees)
# 2. Each tree trained on random subset of data (bagging)
# 3. Each split considers random subset of features
# 4. Final prediction = majority vote (classification)
#                    or average (regression)

# Example:
# Tree 1: "Cat" (70% confident)
# Tree 2: "Dog" (60% confident)
# Tree 3: "Cat" (80% confident)
# Tree 4: "Cat" (55% confident)
# Tree 5: "Dog" (65% confident)
#
# Final: "Cat" wins 3-2!
        </pre>
      </div>

      <h4>Why Random Forests Work</h4>
      <ul>
        <li><strong>Reduces overfitting:</strong> Individual trees might overfit, but averaging cancels out noise</li>
        <li><strong>Handles diverse features:</strong> Different trees focus on different features</li>
        <li><strong>Robust to outliers:</strong> Outliers affect few trees, not the ensemble</li>
        <li><strong>Feature importance:</strong> Shows which features matter most</li>
      </ul>

      <div class="code-block">
        <pre>
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=10,         # Depth of each tree
    min_samples_leaf=5,
    max_features='sqrt',  # sqrt(n_features) per split
    n_jobs=-1             # Use all CPU cores
)

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# Feature importance
for feature, importance in zip(feature_names, rf.feature_importances_):
    print(f"{feature}: {importance:.3f}")
        </pre>
      </div>

      <h3>When to Use Trees vs Neural Networks</h3>
      <table>
        <tr><th>Use Decision Trees/Random Forests</th><th>Use Neural Networks</th></tr>
        <tr><td>Tabular data (spreadsheets)</td><td>Images, audio, text</td></tr>
        <tr><td>Small to medium datasets</td><td>Large datasets (millions)</td></tr>
        <tr><td>Need interpretability</td><td>Accuracy is priority</td></tr>
        <tr><td>Limited compute resources</td><td>GPU available</td></tr>
        <tr><td>Quick baseline needed</td><td>Complex patterns expected</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Decision Trees</strong> learn if-then rules by finding best splits</p>
        <p>2. <strong>Gini impurity</strong> measures how pure a node is</p>
        <p>3. <strong>Overfitting</strong> happens when trees are too deep - limit depth!</p>
        <p>4. <strong>Random Forests</strong> combine many trees for better accuracy</p>
        <p>5. <strong>Feature importance</strong> helps understand what drives predictions</p>
      </div>
    `,
    questions: [
      {
        id: "6-1",
        type: "mcq",
        question: "What does Gini impurity measure in a decision tree?",
        options: ["The depth of the tree", "How well a node separates classes (purity)", "The number of features", "The training time"],
        correctAnswer: "How well a node separates classes (purity)",
        explanation: "Gini impurity measures how 'pure' a node is. A Gini of 0 means all samples are the same class (pure). A Gini of 0.5 means samples are evenly split (maximum impurity). The algorithm finds splits that minimize Gini."
      },
      {
        id: "6-2",
        type: "mcq",
        question: "How does a Random Forest make predictions?",
        options: ["Uses the single best tree", "Averages predictions from many trees trained on random subsets", "Uses only the deepest tree", "Picks the fastest tree"],
        correctAnswer: "Averages predictions from many trees trained on random subsets",
        explanation: "Random Forest trains many trees, each on a random subset of data and features. Final prediction is the majority vote (classification) or average (regression) of all trees. This reduces overfitting."
      },
      {
        id: "6-3",
        type: "descriptive",
        question: "Explain why Random Forests are less prone to overfitting than a single deep decision tree.",
        keywords: ["ensemble", "averaging", "bagging", "random", "subset", "noise", "cancel", "variance", "multiple", "trees"],
        explanation: "Random Forests reduce overfitting by: (1) Training each tree on a random subset of data (bagging) - different trees see different examples. (2) Each split considers random features - trees learn different patterns. (3) Averaging predictions cancels out individual tree errors. A single deep tree memorizes noise, but the ensemble averages it out."
      },
      {
        id: "6-4",
        type: "mcq",
        question: "When should you prefer Random Forests over neural networks?",
        options: ["When working with image data", "When working with tabular data and needing interpretability", "When you have millions of training examples", "When accuracy is the only priority"],
        correctAnswer: "When working with tabular data and needing interpretability",
        explanation: "Random Forests excel on tabular data (spreadsheets), provide feature importance for interpretability, train quickly without GPUs, and work well on smaller datasets. Neural networks are better for images, text, and very large datasets."
      }
    ]
  },
  {
    id: 7,
    title: "Classical ML: Model Evaluation & Metrics",
    description: "Learn to properly evaluate models with precision, recall, F1, and more",
    duration: "50 min",
    difficulty: "Intermediate",
    content: `
      <h2>Why Evaluation Matters</h2>
      <p>A model that's 99% accurate might still be useless. Proper evaluation tells you if your model actually solves the problem.</p>

      <div class="highlight">
        <p><strong>Example:</strong> Fraud detection on 10,000 transactions where only 100 are fraud.</p>
        <p>A model that predicts "not fraud" for everything gets 99% accuracy but catches zero fraud!</p>
      </div>

      <h3>The Confusion Matrix</h3>
      <p>The foundation of classification evaluation:</p>

      <div class="code-block">
        <pre>
                    Predicted
                  Pos    Neg
Actual  Pos  |   TP   |  FN   |
        Neg  |   FP   |  TN   |

TP = True Positive  (correctly predicted positive)
TN = True Negative  (correctly predicted negative)
FP = False Positive (incorrectly predicted positive) - "False Alarm"
FN = False Negative (incorrectly predicted negative) - "Missed"

# Example: Disease diagnosis
# TP = Sick person correctly diagnosed
# TN = Healthy person correctly cleared
# FP = Healthy person incorrectly diagnosed (unnecessary treatment)
# FN = Sick person incorrectly cleared (dangerous!)
        </pre>
      </div>

      <h3>Key Metrics</h3>

      <h4>Accuracy</h4>
      <div class="code-block">
        <pre>
Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Good for balanced datasets
# Misleading for imbalanced datasets (like fraud detection)
        </pre>
      </div>

      <h4>Precision: "Of predicted positives, how many were correct?"</h4>
      <div class="code-block">
        <pre>
Precision = TP / (TP + FP)

# High precision = Few false alarms
# Important when: False positives are costly
# Example: Spam filter - don't want real emails in spam folder
        </pre>
      </div>

      <h4>Recall (Sensitivity): "Of actual positives, how many did we catch?"</h4>
      <div class="code-block">
        <pre>
Recall = TP / (TP + FN)

# High recall = Catch most positives
# Important when: Missing positives is costly
# Example: Cancer detection - don't want to miss any cases
        </pre>
      </div>

      <h4>F1 Score: Balance of Precision and Recall</h4>
      <div class="code-block">
        <pre>
F1 = 2 × (Precision × Recall) / (Precision + Recall)

# Harmonic mean - penalizes extreme imbalance
# Use when you need to balance precision and recall
# Range: 0 to 1 (higher is better)
        </pre>
      </div>

      <h3>Precision-Recall Tradeoff</h3>
      <p>You usually can't maximize both - there's a tradeoff:</p>

      <div class="code-block">
        <pre>
# Adjust classification threshold:

# Threshold = 0.9 (only predict positive if very confident)
# → High Precision, Low Recall
# → Few predictions, but most are correct

# Threshold = 0.1 (predict positive even with low confidence)
# → Low Precision, High Recall
# → Catch most positives, but many false alarms

# Choose threshold based on business needs!
        </pre>
      </div>

      <h3>ROC Curve and AUC</h3>
      <p>Evaluate performance across all thresholds:</p>

      <div class="code-block">
        <pre>
# ROC Curve plots:
# - Y-axis: True Positive Rate (Recall)
# - X-axis: False Positive Rate

# AUC = Area Under ROC Curve
# - AUC = 1.0: Perfect classifier
# - AUC = 0.5: Random guessing
# - AUC < 0.5: Worse than random (flip predictions!)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
        </pre>
      </div>

      <h3>Regression Metrics</h3>

      <h4>Mean Squared Error (MSE)</h4>
      <div class="code-block">
        <pre>
MSE = (1/n) × Σ(predicted - actual)²

# Penalizes large errors heavily (squared)
# Same units as target squared
        </pre>
      </div>

      <h4>Root Mean Squared Error (RMSE)</h4>
      <div class="code-block">
        <pre>
RMSE = √MSE

# Same units as target - easier to interpret
# "On average, predictions are off by RMSE"
        </pre>
      </div>

      <h4>Mean Absolute Error (MAE)</h4>
      <div class="code-block">
        <pre>
MAE = (1/n) × Σ|predicted - actual|

# Less sensitive to outliers than MSE
# Easier to interpret: average absolute error
        </pre>
      </div>

      <h4>R² Score (Coefficient of Determination)</h4>
      <div class="code-block">
        <pre>
R² = 1 - (SS_res / SS_tot)

# How much variance is explained by the model
# R² = 1: Perfect predictions
# R² = 0: Model predicts mean (no learning)
# R² < 0: Worse than predicting mean
        </pre>
      </div>

      <h3>Cross-Validation: Robust Evaluation</h3>
      <div class="code-block">
        <pre>
# Don't trust a single train/test split!

# K-Fold Cross-Validation:
# 1. Split data into K folds (e.g., K=5)
# 2. Train on K-1 folds, test on remaining fold
# 3. Repeat K times, rotating test fold
# 4. Average the scores

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
        </pre>
      </div>

      <h3>Practical Evaluation Code</h3>
      <div class="code-block">
        <pre>
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score
)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Full classification report
print(classification_report(y_test, y_pred))

# Individual metrics
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Accuracy alone</strong> is misleading for imbalanced data</p>
        <p>2. <strong>Precision</strong> = avoiding false alarms (important for spam)</p>
        <p>3. <strong>Recall</strong> = catching all positives (important for disease detection)</p>
        <p>4. <strong>F1</strong> balances precision and recall</p>
        <p>5. <strong>Cross-validation</strong> gives robust estimates</p>
      </div>
    `,
    questions: [
      {
        id: "7-1",
        type: "mcq",
        question: "For a cancer detection model, which metric is most important?",
        options: ["Accuracy", "Precision", "Recall", "Training speed"],
        correctAnswer: "Recall",
        explanation: "For cancer detection, missing a positive case (false negative) is extremely dangerous - the patient won't get treatment. High recall ensures we catch as many cases as possible, even if some healthy people need follow-up tests (false positives)."
      },
      {
        id: "7-2",
        type: "mcq",
        question: "What does a model with high precision but low recall indicate?",
        options: ["It makes few predictions but they're mostly correct", "It catches all positives but has many false alarms", "It's perfect", "It's random guessing"],
        correctAnswer: "It makes few predictions but they're mostly correct",
        explanation: "High precision, low recall means: when the model predicts positive, it's usually right (few false positives), BUT it misses many actual positives (many false negatives). It's conservative - only predicts positive when very confident."
      },
      {
        id: "7-3",
        type: "descriptive",
        question: "Explain why accuracy can be misleading and what metrics you would use instead for an imbalanced dataset.",
        keywords: ["imbalanced", "precision", "recall", "F1", "confusion matrix", "false positive", "false negative", "majority class", "minority"],
        explanation: "Accuracy is misleading when classes are imbalanced: a model predicting only the majority class achieves high accuracy but fails on the minority class (e.g., 99% accuracy by always predicting 'not fraud' while catching 0% of fraud). Instead, use: (1) Precision and Recall to understand error types, (2) F1 score for a balanced metric, (3) AUC to evaluate across thresholds, (4) Confusion matrix to see all error types."
      },
      {
        id: "7-4",
        type: "mcq",
        question: "What does an AUC score of 0.5 indicate?",
        options: ["Perfect classifier", "Good classifier", "Random guessing", "Inverse predictions"],
        correctAnswer: "Random guessing",
        explanation: "AUC = 0.5 means the model performs no better than random guessing. AUC = 1.0 is perfect classification. AUC < 0.5 means the model is worse than random (flip the predictions to improve)."
      }
    ]
  },
  {
    id: 8,
    title: "Data Science: Cleaning & Feature Engineering",
    description: "Learn to prepare real-world messy data for machine learning",
    duration: "55 min",
    difficulty: "Intermediate",
    content: `
      <h2>The Data Reality</h2>
      <p>Real-world data is messy. Data scientists spend 60-80% of their time on data preparation. Good data preparation often matters more than algorithm choice!</p>

      <h3>Common Data Problems</h3>

      <h4>1. Missing Values</h4>
      <div class="code-block">
        <pre>
import pandas as pd
import numpy as np

# Check for missing values
df.isnull().sum()

# Strategies for handling missing values:

# 1. Drop rows with missing values (if few)
df.dropna()

# 2. Fill with mean/median (numerical)
df['age'].fillna(df['age'].median(), inplace=True)

# 3. Fill with mode (categorical)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# 4. Fill with a flag value
df['value'].fillna(-999, inplace=True)

# 5. Forward/backward fill (time series)
df['price'].fillna(method='ffill', inplace=True)

# 6. Advanced: Use ML to predict missing values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
        </pre>
      </div>

      <h4>2. Outliers</h4>
      <div class="code-block">
        <pre>
# Detect outliers using IQR
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]

# Handle outliers:
# 1. Remove them (if errors)
df = df[df['value'].between(lower_bound, upper_bound)]

# 2. Cap them (winsorization)
df['value'] = df['value'].clip(lower_bound, upper_bound)

# 3. Transform (log transform reduces outlier impact)
df['value_log'] = np.log1p(df['value'])
        </pre>
      </div>

      <h4>3. Inconsistent Data</h4>
      <div class="code-block">
        <pre>
# Standardize text
df['city'] = df['city'].str.lower().str.strip()

# Fix inconsistent categories
# Before: ['New York', 'new york', 'NEW YORK', 'NY']
# After: ['new_york', 'new_york', 'new_york', 'new_york']

mapping = {
    'new york': 'new_york',
    'ny': 'new_york',
    'n.y.': 'new_york'
}
df['city'] = df['city'].replace(mapping)

# Fix data types
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')
        </pre>
      </div>

      <h3>Feature Engineering</h3>
      <p>Creating new features from existing data often improves model performance dramatically.</p>

      <h4>Numerical Features</h4>
      <div class="code-block">
        <pre>
# Scaling: Make features comparable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 (good for most ML)
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# MinMaxScaler: range [0, 1] (good for neural networks)
scaler = MinMaxScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Log transform: Handle skewed distributions
df['income_log'] = np.log1p(df['income'])

# Binning: Convert continuous to categorical
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 35, 50, 100],
    labels=['youth', 'young_adult', 'middle', 'senior'])

# Polynomial features: Capture non-linear relationships
df['age_squared'] = df['age'] ** 2
df['age_income'] = df['age'] * df['income']
        </pre>
      </div>

      <h4>Categorical Features</h4>
      <div class="code-block">
        <pre>
# One-Hot Encoding: For nominal categories (no order)
df = pd.get_dummies(df, columns=['city'])
# city → city_new_york, city_boston, city_chicago

# Label Encoding: For ordinal categories (has order)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
# ['high school', 'bachelor', 'master'] → [0, 1, 2]

# Target Encoding: Replace category with mean of target
# Good for high-cardinality features
city_means = df.groupby('city')['price'].mean()
df['city_encoded'] = df['city'].map(city_means)
        </pre>
      </div>

      <h4>Date/Time Features</h4>
      <div class="code-block">
        <pre>
# Extract useful components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['hour'] = df['datetime'].dt.hour
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

# Days since event
df['days_since_signup'] = (df['purchase_date'] - df['signup_date']).dt.days

# Cyclical encoding for periodic features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        </pre>
      </div>

      <h4>Text Features (Basic)</h4>
      <div class="code-block">
        <pre>
# Simple text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_question'] = df['text'].str.contains('\\?').astype(int)

# TF-IDF for text classification
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(df['text'])
        </pre>
      </div>

      <h3>Feature Selection</h3>
      <p>Too many features can hurt performance. Select the most important ones:</p>

      <div class="code-block">
        <pre>
# Method 1: Correlation with target
correlations = df.corr()['target'].abs().sort_values(descending=True)

# Method 2: Feature importance from Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(10)

# Method 3: Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Method 4: Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=rf, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
        </pre>
      </div>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>Data cleaning</strong> handles missing values, outliers, and inconsistencies</p>
        <p>2. <strong>Feature scaling</strong> makes numerical features comparable</p>
        <p>3. <strong>Encoding</strong> converts categories to numbers ML can use</p>
        <p>4. <strong>Feature engineering</strong> creates new predictive features</p>
        <p>5. <strong>Feature selection</strong> keeps only useful features</p>
      </div>
    `,
    questions: [
      {
        id: "8-1",
        type: "mcq",
        question: "Why is log transformation useful for skewed data?",
        options: ["It makes data negative", "It compresses large values and spreads small values, reducing outlier impact", "It removes missing values", "It converts to categories"],
        correctAnswer: "It compresses large values and spreads small values, reducing outlier impact",
        explanation: "Log transformation compresses the range of large values while spreading out small values. This reduces the impact of outliers and makes highly skewed distributions more normal, which many algorithms prefer."
      },
      {
        id: "8-2",
        type: "mcq",
        question: "When should you use one-hot encoding vs label encoding for categorical features?",
        options: ["Always use one-hot", "One-hot for nominal (no order), label for ordinal (has order)", "Always use label encoding", "They're identical"],
        correctAnswer: "One-hot for nominal (no order), label for ordinal (has order)",
        explanation: "One-hot encoding for nominal categories (city, color) because there's no inherent order. Label encoding for ordinal categories (education level, satisfaction rating) because the numbers preserve the ordering relationship."
      },
      {
        id: "8-3",
        type: "descriptive",
        question: "Describe three strategies for handling missing values and when to use each.",
        keywords: ["drop", "mean", "median", "mode", "impute", "flag", "forward fill", "KNN", "prediction", "missing"],
        explanation: "Strategies: (1) Drop rows - when few missing values and data is large enough. (2) Fill with mean/median - for numerical data when missing at random; median is robust to outliers. (3) Fill with mode - for categorical data. (4) Forward/backward fill - for time series where previous value is a good estimate. (5) ML imputation (KNN) - when missing values have patterns related to other features."
      },
      {
        id: "8-4",
        type: "mcq",
        question: "What is the purpose of feature scaling (standardization)?",
        options: ["To remove features", "To make features comparable and help algorithms converge faster", "To increase dataset size", "To remove outliers"],
        correctAnswer: "To make features comparable and help algorithms converge faster",
        explanation: "Feature scaling ensures all features are on similar scales. Without scaling, features with large ranges (like income: 0-1M) dominate features with small ranges (like age: 0-100). Scaling helps gradient-based algorithms converge faster and makes distance-based algorithms work correctly."
      }
    ]
  },
  {
    id: 9,
    title: "Reinforcement Learning Fundamentals",
    description: "Learn how agents learn through trial and error to maximize rewards",
    duration: "60 min",
    difficulty: "Intermediate",
    content: `
      <h2>What is Reinforcement Learning?</h2>
      <p>Reinforcement Learning (RL) is learning through interaction. An agent takes actions in an environment, receives rewards or penalties, and learns to maximize long-term rewards.</p>

      <div class="highlight">
        <p><strong>Key difference from supervised learning:</strong></p>
        <ul>
          <li>Supervised: Learn from labeled examples (input → correct output)</li>
          <li>Reinforcement: Learn from rewards (action → reward signal)</li>
        </ul>
      </div>

      <h3>The RL Framework</h3>
      <div class="code-block">
        <pre>
# Core components:
# - Agent: The learner/decision maker
# - Environment: What the agent interacts with
# - State (s): Current situation
# - Action (a): What the agent can do
# - Reward (r): Feedback signal (positive or negative)
# - Policy (π): Strategy for choosing actions

# The loop:
# 1. Agent observes state s
# 2. Agent chooses action a based on policy π
# 3. Environment transitions to new state s'
# 4. Agent receives reward r
# 5. Agent updates policy to get better rewards
# 6. Repeat
        </pre>
      </div>

      <h3>Real-World Examples</h3>
      <ul>
        <li><strong>Game AI:</strong> State = game screen, Action = controller input, Reward = score</li>
        <li><strong>Robotics:</strong> State = sensor readings, Action = motor commands, Reward = task completion</li>
        <li><strong>Recommendation:</strong> State = user history, Action = recommend item, Reward = user clicks</li>
        <li><strong>Trading:</strong> State = market data, Action = buy/sell/hold, Reward = profit</li>
      </ul>

      <h3>The Exploration-Exploitation Tradeoff</h3>
      <p>The fundamental dilemma in RL:</p>

      <div class="code-block">
        <pre>
# Exploitation: Do what you know works best
# - Use current best action
# - Maximize immediate reward
# - Risk: Might miss better options

# Exploration: Try new things
# - Take random or uncertain actions
# - Learn about environment
# - Risk: Might get poor rewards

# Solution: ε-greedy strategy
def choose_action(state, epsilon=0.1):
    if random() < epsilon:
        return random_action()  # Explore (10% of time)
    else:
        return best_known_action(state)  # Exploit (90% of time)
        </pre>
      </div>

      <h3>Value Functions</h3>
      <p>Value functions estimate how good states or actions are:</p>

      <h4>State Value V(s)</h4>
      <div class="code-block">
        <pre>
# V(s) = Expected total reward starting from state s

# Example: Chess position value
# V(winning position) ≈ 1.0  (high value - likely to win)
# V(losing position) ≈ 0.0  (low value - likely to lose)
# V(equal position) ≈ 0.5  (uncertain)
        </pre>
      </div>

      <h4>Action Value Q(s, a)</h4>
      <div class="code-block">
        <pre>
# Q(s, a) = Expected total reward from taking action a in state s

# Example: Q-table for simple game
#              | Action: Left | Action: Right |
# State A      |     10       |      5        |
# State B      |      3       |      8        |

# In State A, choose Left (Q=10 > 5)
# In State B, choose Right (Q=8 > 3)
        </pre>
      </div>

      <h3>Q-Learning: A Classic Algorithm</h3>
      <div class="code-block">
        <pre>
# Q-Learning update rule:
# Q(s,a) ← Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]

# Where:
# α = learning rate (how fast to update)
# γ = discount factor (how much to value future rewards)
# r = immediate reward
# s' = next state
# max(Q(s',a')) = best Q-value in next state

def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(float)  # Q-table initialized to 0

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random() < epsilon:
                action = env.random_action()
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            # Take action, observe result
            next_state, reward, done = env.step(action)

            # Q-learning update
            best_next_q = max(Q[(next_state, a)] for a in actions)
            Q[(state, action)] += alpha * (
                reward + gamma * best_next_q - Q[(state, action)]
            )

            state = next_state

    return Q
        </pre>
      </div>

      <h3>Deep Reinforcement Learning</h3>
      <p>When state space is too large for Q-tables, use neural networks:</p>

      <div class="code-block">
        <pre>
# Deep Q-Network (DQN):
# Instead of Q-table, use neural network to approximate Q-values

# Input: State (e.g., game pixels)
# Output: Q-value for each action

# Key innovations:
# 1. Experience Replay: Store transitions, sample randomly
# 2. Target Network: Stabilize training with separate network
# 3. Frame Stacking: Input multiple frames for motion

# DQN achieved superhuman performance on Atari games!
        </pre>
      </div>

      <h3>Policy Gradient Methods</h3>
      <p>Instead of learning values, directly learn the policy:</p>

      <div class="code-block">
        <pre>
# Policy network outputs action probabilities directly
# π(a|s) = probability of action a given state s

# Update: Increase probability of actions that led to good rewards
# Decrease probability of actions that led to bad rewards

# Advantage: Can handle continuous action spaces
# Disadvantage: High variance, needs many samples
        </pre>
      </div>

      <h3>Applications of RL</h3>
      <table>
        <tr><th>Application</th><th>Notable Achievement</th></tr>
        <tr><td>Game Playing</td><td>AlphaGo, AlphaStar, OpenAI Five</td></tr>
        <tr><td>Robotics</td><td>Robot manipulation, locomotion</td></tr>
        <tr><td>LLM Training</td><td>RLHF (Reinforcement Learning from Human Feedback)</td></tr>
        <tr><td>Autonomous Driving</td><td>Decision making in complex traffic</td></tr>
        <tr><td>Resource Management</td><td>Data center cooling (Google)</td></tr>
      </table>

      <h3>Key Takeaways</h3>
      <div class="highlight">
        <p>1. <strong>RL</strong> learns from rewards through trial and error</p>
        <p>2. <strong>Exploration vs exploitation</strong> balances trying new things vs using known good actions</p>
        <p>3. <strong>Value functions</strong> estimate how good states/actions are</p>
        <p>4. <strong>Q-learning</strong> learns action values without a model of the environment</p>
        <p>5. <strong>Deep RL</strong> uses neural networks for complex state spaces</p>
      </div>
    `,
    questions: [
      {
        id: "9-1",
        type: "mcq",
        question: "What is the exploration-exploitation tradeoff?",
        options: ["Training vs testing", "Balancing trying new actions vs using known good actions", "Speed vs accuracy", "Memory vs computation"],
        correctAnswer: "Balancing trying new actions vs using known good actions",
        explanation: "The exploration-exploitation tradeoff: exploitation means using actions you know work well (maximize immediate reward), exploration means trying new actions to potentially find better strategies. ε-greedy balances this by exploring randomly ε% of the time."
      },
      {
        id: "9-2",
        type: "mcq",
        question: "What does Q(s, a) represent in Q-learning?",
        options: ["Quality of the neural network", "Expected total reward from taking action a in state s", "Query response time", "Quantity of training data"],
        correctAnswer: "Expected total reward from taking action a in state s",
        explanation: "Q(s, a) is the action-value function - it estimates the expected cumulative future reward if you take action a in state s and then follow the optimal policy afterward."
      },
      {
        id: "9-3",
        type: "descriptive",
        question: "Explain how reinforcement learning differs from supervised learning, with an example.",
        keywords: ["reward", "label", "interaction", "environment", "trial", "error", "feedback", "delayed", "agent", "action"],
        explanation: "Key differences: (1) Supervised learning has labeled examples (input→output), RL has rewards from environment interaction. (2) RL feedback is delayed - you might not know if an action was good until many steps later. (3) RL agent's actions affect future states. Example: Training a robot to walk - no 'correct answer' labels exist, but it learns from falling (negative reward) and moving forward (positive reward)."
      },
      {
        id: "9-4",
        type: "mcq",
        question: "What is RLHF used for in modern AI?",
        options: ["Image generation", "Training large language models to be more helpful and harmless", "Autonomous driving only", "Database optimization"],
        correctAnswer: "Training large language models to be more helpful and harmless",
        explanation: "RLHF (Reinforcement Learning from Human Feedback) is used to fine-tune LLMs like ChatGPT and Claude. Human raters compare model outputs, and RL optimizes the model to produce outputs humans prefer - making it more helpful, harmless, and honest."
      }
    ]
  }
];

export default foundationalLessons;
