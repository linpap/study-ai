import { PracticeExercise } from '@/types/practice';

export const practiceExercises: PracticeExercise[] = [
  {
    id: 1,
    title: 'Implement Dot Product',
    description: 'Calculate the dot product of two vectors - a fundamental operation in neural networks.',
    difficulty: 'beginner',
    category: 'Linear Algebra',
    estimatedTime: '10 min',
    problemStatement: `The **dot product** is essential in AI - it's used in neural network layers, attention mechanisms, and similarity calculations.

Write a function called \`dotProduct\` that takes two arrays of equal length and returns their dot product.

**Formula:** a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

**Example:**
\`\`\`javascript
dotProduct([1, 2, 3], [4, 5, 6]) // returns 1×4 + 2×5 + 3×6 = 32
dotProduct([1, 0], [0, 1]) // returns 0 (perpendicular vectors)
dotProduct([2, 2], [2, 2]) // returns 8
\`\`\``,
    hints: [
      'Loop through both arrays simultaneously',
      'Multiply corresponding elements and accumulate the sum',
      'The result is a single number, not an array',
    ],
    language: 'javascript',
    starterCode: `function dotProduct(a, b) {
  // Your code here
  // Multiply corresponding elements and sum them
}`,
    solutionCode: `function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}`,
    testCases: [
      { id: '1', description: 'dotProduct([1, 2, 3], [4, 5, 6]) should return 32', input: 'dotProduct([1, 2, 3], [4, 5, 6])', expectedOutput: '32' },
      { id: '2', description: 'dotProduct([1, 0], [0, 1]) should return 0', input: 'dotProduct([1, 0], [0, 1])', expectedOutput: '0' },
      { id: '3', description: 'dotProduct([2, 2], [2, 2]) should return 8', input: 'dotProduct([2, 2], [2, 2])', expectedOutput: '8' },
      { id: '4', description: 'dotProduct([-1, 2], [3, 4]) should return 5', input: 'dotProduct([-1, 2], [3, 4])', expectedOutput: '5', isHidden: true },
    ],
    tags: ['linear algebra', 'vectors', 'neural networks'],
  },
  {
    id: 2,
    title: 'Implement Sigmoid Activation',
    description: 'Build the classic sigmoid activation function used in neural networks.',
    difficulty: 'beginner',
    category: 'Neural Networks',
    estimatedTime: '10 min',
    problemStatement: `The **sigmoid function** squashes any input to a value between 0 and 1, making it useful for probability outputs.

**Formula:** σ(x) = 1 / (1 + e^(-x))

Write a function called \`sigmoid\` that applies the sigmoid function to a number.

**Example:**
\`\`\`javascript
sigmoid(0)   // returns 0.5 (exactly in the middle)
sigmoid(10)  // returns ~0.99995 (close to 1)
sigmoid(-10) // returns ~0.00005 (close to 0)
\`\`\``,
    hints: [
      'Use Math.exp() for the exponential function',
      'Remember the formula: 1 / (1 + e^(-x))',
      'Large positive x → close to 1, large negative x → close to 0',
    ],
    language: 'javascript',
    starterCode: `function sigmoid(x) {
  // Return 1 / (1 + e^(-x))
}`,
    solutionCode: `function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}`,
    testCases: [
      { id: '1', description: 'sigmoid(0) should return 0.5', input: 'sigmoid(0)', expectedOutput: '0.5' },
      { id: '2', description: 'sigmoid(10) should be close to 1', input: 'Math.round(sigmoid(10) * 100000) / 100000', expectedOutput: '0.99995' },
      { id: '3', description: 'sigmoid(-10) should be close to 0', input: 'Math.round(sigmoid(-10) * 100000) / 100000', expectedOutput: '0.00005' },
      { id: '4', description: 'sigmoid(2) should be approximately 0.88', input: 'Math.round(sigmoid(2) * 100) / 100', expectedOutput: '0.88', isHidden: true },
    ],
    tags: ['activation functions', 'neural networks', 'math'],
  },
  {
    id: 3,
    title: 'Implement ReLU Activation',
    description: 'Build the ReLU activation function - the most popular activation in deep learning.',
    difficulty: 'beginner',
    category: 'Neural Networks',
    estimatedTime: '5 min',
    problemStatement: `**ReLU (Rectified Linear Unit)** is the most widely used activation function in modern neural networks due to its simplicity and effectiveness.

**Formula:** ReLU(x) = max(0, x)

Write a function called \`relu\` that returns 0 for negative inputs and the input itself for non-negative inputs.

**Example:**
\`\`\`javascript
relu(5)  // returns 5
relu(-3) // returns 0
relu(0)  // returns 0
\`\`\``,
    hints: [
      'Return 0 if x is negative, otherwise return x',
      'You can use Math.max(0, x)',
      'Or use a simple conditional: x > 0 ? x : 0',
    ],
    language: 'javascript',
    starterCode: `function relu(x) {
  // Return max(0, x)
}`,
    solutionCode: `function relu(x) {
  return Math.max(0, x);
}`,
    testCases: [
      { id: '1', description: 'relu(5) should return 5', input: 'relu(5)', expectedOutput: '5' },
      { id: '2', description: 'relu(-3) should return 0', input: 'relu(-3)', expectedOutput: '0' },
      { id: '3', description: 'relu(0) should return 0', input: 'relu(0)', expectedOutput: '0' },
      { id: '4', description: 'relu(-0.5) should return 0', input: 'relu(-0.5)', expectedOutput: '0', isHidden: true },
    ],
    tags: ['activation functions', 'neural networks', 'deep learning'],
  },
  {
    id: 4,
    title: 'Implement Softmax Function',
    description: 'Convert raw scores into probabilities - essential for classification.',
    difficulty: 'intermediate',
    category: 'Neural Networks',
    estimatedTime: '15 min',
    problemStatement: `**Softmax** converts a vector of raw scores (logits) into a probability distribution that sums to 1.

**Formula:** softmax(xᵢ) = e^(xᵢ) / Σe^(xⱼ)

Write a function called \`softmax\` that takes an array of numbers and returns probabilities.

**Example:**
\`\`\`javascript
softmax([1, 2, 3])
// returns approximately [0.09, 0.24, 0.67]
// (higher inputs get higher probabilities)

softmax([0, 0, 0])
// returns [0.333, 0.333, 0.333]
// (equal inputs = equal probabilities)
\`\`\``,
    hints: [
      'First, compute e^x for each element',
      'Then, sum all the exponentials',
      'Finally, divide each exponential by the sum',
      'For numerical stability, subtract max(x) from all elements first',
    ],
    language: 'javascript',
    starterCode: `function softmax(arr) {
  // 1. Compute exponentials
  // 2. Sum them
  // 3. Divide each by sum
}`,
    solutionCode: `function softmax(arr) {
  // Subtract max for numerical stability
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}`,
    testCases: [
      { id: '1', description: 'softmax([0, 0, 0]) should return equal probabilities', input: 'softmax([0, 0, 0]).map(x => Math.round(x * 100) / 100).join(",")', expectedOutput: '0.33,0.33,0.33' },
      { id: '2', description: 'softmax probabilities should sum to 1', input: 'Math.round(softmax([1, 2, 3]).reduce((a, b) => a + b, 0))', expectedOutput: '1' },
      { id: '3', description: 'Higher inputs should get higher probabilities', input: 'softmax([1, 2, 3])[2] > softmax([1, 2, 3])[0]', expectedOutput: 'true' },
      { id: '4', description: 'softmax([10, 0, 0]) first element should be close to 1', input: 'Math.round(softmax([10, 0, 0])[0] * 100) / 100', expectedOutput: '1', isHidden: true },
    ],
    tags: ['probability', 'classification', 'neural networks'],
  },
  {
    id: 5,
    title: 'Calculate Mean Squared Error',
    description: 'Implement the most common loss function for regression.',
    difficulty: 'beginner',
    category: 'Model Evaluation',
    estimatedTime: '10 min',
    problemStatement: `**Mean Squared Error (MSE)** measures how far predictions are from actual values - the loss function for regression.

**Formula:** MSE = (1/n) × Σ(predicted - actual)²

Write a function called \`mse\` that takes arrays of predictions and actual values.

**Example:**
\`\`\`javascript
mse([1, 2, 3], [1, 2, 3]) // returns 0 (perfect predictions)
mse([1, 2, 3], [2, 3, 4]) // returns 1 (each off by 1)
mse([0, 0], [1, -1])      // returns 1
\`\`\``,
    hints: [
      'Calculate difference for each pair',
      'Square each difference',
      'Take the average of squared differences',
    ],
    language: 'javascript',
    starterCode: `function mse(predicted, actual) {
  // Calculate mean squared error
}`,
    solutionCode: `function mse(predicted, actual) {
  const n = predicted.length;
  let sumSquaredError = 0;
  for (let i = 0; i < n; i++) {
    const error = predicted[i] - actual[i];
    sumSquaredError += error * error;
  }
  return sumSquaredError / n;
}`,
    testCases: [
      { id: '1', description: 'Perfect predictions should return 0', input: 'mse([1, 2, 3], [1, 2, 3])', expectedOutput: '0' },
      { id: '2', description: 'mse([1, 2, 3], [2, 3, 4]) should return 1', input: 'mse([1, 2, 3], [2, 3, 4])', expectedOutput: '1' },
      { id: '3', description: 'mse([0, 0], [1, -1]) should return 1', input: 'mse([0, 0], [1, -1])', expectedOutput: '1' },
      { id: '4', description: 'mse([0, 0, 0], [3, 0, 0]) should return 3', input: 'mse([0, 0, 0], [3, 0, 0])', expectedOutput: '3', isHidden: true },
    ],
    tags: ['loss functions', 'regression', 'evaluation'],
  },
  {
    id: 6,
    title: 'Implement Accuracy Score',
    description: 'Calculate classification accuracy - the simplest evaluation metric.',
    difficulty: 'beginner',
    category: 'Model Evaluation',
    estimatedTime: '10 min',
    problemStatement: `**Accuracy** is the simplest classification metric: the fraction of correct predictions.

**Formula:** Accuracy = (correct predictions) / (total predictions)

Write a function called \`accuracy\` that takes arrays of predictions and actual labels.

**Example:**
\`\`\`javascript
accuracy([1, 1, 0, 0], [1, 1, 0, 0]) // returns 1.0 (all correct)
accuracy([1, 1, 1, 1], [1, 0, 1, 0]) // returns 0.5 (half correct)
accuracy([0, 0, 0], [1, 1, 1])       // returns 0 (all wrong)
\`\`\``,
    hints: [
      'Count how many predictions match actual values',
      'Divide by total number of predictions',
      'Return a number between 0 and 1',
    ],
    language: 'javascript',
    starterCode: `function accuracy(predicted, actual) {
  // Count correct predictions / total predictions
}`,
    solutionCode: `function accuracy(predicted, actual) {
  let correct = 0;
  for (let i = 0; i < predicted.length; i++) {
    if (predicted[i] === actual[i]) {
      correct++;
    }
  }
  return correct / predicted.length;
}`,
    testCases: [
      { id: '1', description: 'All correct should return 1', input: 'accuracy([1, 1, 0, 0], [1, 1, 0, 0])', expectedOutput: '1' },
      { id: '2', description: 'Half correct should return 0.5', input: 'accuracy([1, 1, 1, 1], [1, 0, 1, 0])', expectedOutput: '0.5' },
      { id: '3', description: 'All wrong should return 0', input: 'accuracy([0, 0, 0], [1, 1, 1])', expectedOutput: '0' },
      { id: '4', description: '3 out of 4 correct should return 0.75', input: 'accuracy([1, 1, 1, 0], [1, 1, 1, 1])', expectedOutput: '0.75', isHidden: true },
    ],
    tags: ['classification', 'evaluation', 'metrics'],
  },
  {
    id: 7,
    title: 'Normalize a Vector',
    description: 'Scale a vector to unit length - essential for embeddings and similarity.',
    difficulty: 'intermediate',
    category: 'Linear Algebra',
    estimatedTime: '15 min',
    problemStatement: `**Normalization** scales a vector to have length 1 (unit vector). This is crucial for:
- Comparing embeddings
- Cosine similarity
- Stable neural network training

**Formula:** normalized = vector / ||vector||

Where ||vector|| is the L2 norm (Euclidean length): sqrt(x₁² + x₂² + ... + xₙ²)

**Example:**
\`\`\`javascript
normalize([3, 4])  // returns [0.6, 0.8] (length becomes 1)
normalize([1, 0])  // returns [1, 0] (already unit length)
normalize([2, 2, 1]) // returns [0.667, 0.667, 0.333]
\`\`\``,
    hints: [
      'First calculate the L2 norm: sqrt of sum of squares',
      'Then divide each element by the norm',
      'Handle the edge case where norm is 0',
    ],
    language: 'javascript',
    starterCode: `function normalize(vector) {
  // 1. Calculate L2 norm (length)
  // 2. Divide each element by norm
}`,
    solutionCode: `function normalize(vector) {
  // Calculate L2 norm
  const sumSquares = vector.reduce((sum, x) => sum + x * x, 0);
  const norm = Math.sqrt(sumSquares);

  // Handle zero vector
  if (norm === 0) return vector;

  // Divide each element by norm
  return vector.map(x => x / norm);
}`,
    testCases: [
      { id: '1', description: 'normalize([3, 4]) should return [0.6, 0.8]', input: 'normalize([3, 4]).map(x => Math.round(x * 10) / 10).join(",")', expectedOutput: '0.6,0.8' },
      { id: '2', description: 'Normalized vector should have length 1', input: 'Math.round(Math.sqrt(normalize([3, 4]).reduce((s, x) => s + x*x, 0)))', expectedOutput: '1' },
      { id: '3', description: 'normalize([1, 0]) should return [1, 0]', input: 'normalize([1, 0]).join(",")', expectedOutput: '1,0' },
      { id: '4', description: 'normalize([0, 0]) should handle zero vector', input: 'normalize([0, 0]).join(",")', expectedOutput: '0,0', isHidden: true },
    ],
    tags: ['linear algebra', 'embeddings', 'normalization'],
  },
  {
    id: 8,
    title: 'Implement Cosine Similarity',
    description: 'Measure similarity between vectors - used in embeddings and search.',
    difficulty: 'intermediate',
    category: 'Linear Algebra',
    estimatedTime: '15 min',
    problemStatement: `**Cosine similarity** measures how similar two vectors are regardless of their magnitude.

**Formula:** cos(θ) = (A · B) / (||A|| × ||B||)

Range: -1 (opposite) to 1 (identical direction)

**Use cases:** Document similarity, recommendation systems, semantic search

**Example:**
\`\`\`javascript
cosineSimilarity([1, 0], [1, 0])   // returns 1 (identical)
cosineSimilarity([1, 0], [0, 1])   // returns 0 (perpendicular)
cosineSimilarity([1, 0], [-1, 0])  // returns -1 (opposite)
cosineSimilarity([1, 2], [2, 4])   // returns 1 (same direction)
\`\`\``,
    hints: [
      'Calculate dot product of the two vectors',
      'Calculate magnitude (L2 norm) of each vector',
      'Divide dot product by product of magnitudes',
    ],
    language: 'javascript',
    starterCode: `function cosineSimilarity(a, b) {
  // cos(θ) = (a · b) / (||a|| × ||b||)
}`,
    solutionCode: `function cosineSimilarity(a, b) {
  // Dot product
  let dotProduct = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
  }

  // Magnitudes
  const magA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const magB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));

  // Handle zero vectors
  if (magA === 0 || magB === 0) return 0;

  return dotProduct / (magA * magB);
}`,
    testCases: [
      { id: '1', description: 'Identical vectors should return 1', input: 'cosineSimilarity([1, 0], [1, 0])', expectedOutput: '1' },
      { id: '2', description: 'Perpendicular vectors should return 0', input: 'cosineSimilarity([1, 0], [0, 1])', expectedOutput: '0' },
      { id: '3', description: 'Opposite vectors should return -1', input: 'cosineSimilarity([1, 0], [-1, 0])', expectedOutput: '-1' },
      { id: '4', description: 'Same direction different magnitude should return 1', input: 'Math.round(cosineSimilarity([1, 2], [2, 4]))', expectedOutput: '1', isHidden: true },
    ],
    tags: ['similarity', 'embeddings', 'search'],
  },
  {
    id: 9,
    title: 'Implement Gradient Descent Step',
    description: 'Build one step of the core optimization algorithm in deep learning.',
    difficulty: 'intermediate',
    category: 'Optimization',
    estimatedTime: '20 min',
    problemStatement: `**Gradient descent** is how neural networks learn. Each step updates parameters to reduce loss.

**Update rule:** new_params = old_params - learning_rate × gradient

Write a function that performs one gradient descent step for a simple linear regression: y = wx + b

**Given:**
- Current weight w and bias b
- Training data points (x, y)
- Learning rate

**Gradients for MSE loss:**
- ∂L/∂w = (2/n) × Σ(wx + b - y) × x
- ∂L/∂b = (2/n) × Σ(wx + b - y)

**Example:**
\`\`\`javascript
gradientDescentStep(
  { w: 0, b: 0 },           // initial params
  [[1, 2], [2, 4], [3, 6]], // data: [x, y] pairs
  0.1                        // learning rate
)
// Returns updated { w, b } closer to true values
\`\`\``,
    hints: [
      'Calculate predictions: wx + b for each x',
      'Calculate errors: prediction - actual y',
      'Calculate gradients using the formulas',
      'Update: new_param = old_param - lr × gradient',
    ],
    language: 'javascript',
    starterCode: `function gradientDescentStep(params, data, learningRate) {
  let { w, b } = params;
  const n = data.length;

  // Calculate gradients
  let gradW = 0;
  let gradB = 0;

  // Your code: compute gradients

  // Update parameters
  // Your code: apply gradient descent update

  return { w, b };
}`,
    solutionCode: `function gradientDescentStep(params, data, learningRate) {
  let { w, b } = params;
  const n = data.length;

  // Calculate gradients
  let gradW = 0;
  let gradB = 0;

  for (const [x, y] of data) {
    const prediction = w * x + b;
    const error = prediction - y;
    gradW += (2 / n) * error * x;
    gradB += (2 / n) * error;
  }

  // Update parameters
  w = w - learningRate * gradW;
  b = b - learningRate * gradB;

  return { w, b };
}`,
    testCases: [
      { id: '1', description: 'Should update weights', input: '(() => { const r = gradientDescentStep({w:0, b:0}, [[1,2],[2,4]], 0.1); return r.w !== 0; })()', expectedOutput: 'true' },
      { id: '2', description: 'Perfect params should not change much', input: '(() => { const r = gradientDescentStep({w:2, b:0}, [[1,2],[2,4]], 0.01); return Math.abs(r.w - 2) < 0.1; })()', expectedOutput: 'true' },
      { id: '3', description: 'Should move toward correct solution', input: '(() => { let p = {w:0, b:0}; for(let i=0; i<100; i++) p = gradientDescentStep(p, [[1,2],[2,4],[3,6]], 0.1); return Math.round(p.w); })()', expectedOutput: '2' },
    ],
    tags: ['optimization', 'gradient descent', 'training'],
  },
  {
    id: 10,
    title: 'Implement K-Nearest Neighbors',
    description: 'Build a simple but powerful classification algorithm from scratch.',
    difficulty: 'advanced',
    category: 'Machine Learning',
    estimatedTime: '25 min',
    problemStatement: `**K-Nearest Neighbors (KNN)** classifies a point based on the majority class of its k nearest neighbors.

**Algorithm:**
1. Calculate distance from new point to all training points
2. Find the k closest training points
3. Return the most common class among those k points

**Example:**
\`\`\`javascript
const trainingData = [
  { point: [0, 0], label: 'A' },
  { point: [1, 1], label: 'A' },
  { point: [5, 5], label: 'B' },
  { point: [6, 6], label: 'B' },
];

knn(trainingData, [0.5, 0.5], 3) // returns 'A' (closer to A points)
knn(trainingData, [5.5, 5.5], 3) // returns 'B' (closer to B points)
\`\`\``,
    hints: [
      'Use Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)',
      'Sort training data by distance to the new point',
      'Take the first k elements after sorting',
      'Count labels and return the most common one',
    ],
    language: 'javascript',
    starterCode: `function knn(trainingData, newPoint, k) {
  // 1. Calculate distances to all training points
  // 2. Sort by distance
  // 3. Take k nearest
  // 4. Return most common label
}`,
    solutionCode: `function knn(trainingData, newPoint, k) {
  // Calculate distances
  const withDistances = trainingData.map(item => {
    let sumSquares = 0;
    for (let i = 0; i < newPoint.length; i++) {
      sumSquares += Math.pow(item.point[i] - newPoint[i], 2);
    }
    return { ...item, distance: Math.sqrt(sumSquares) };
  });

  // Sort by distance and take k nearest
  withDistances.sort((a, b) => a.distance - b.distance);
  const kNearest = withDistances.slice(0, k);

  // Count labels
  const counts = {};
  for (const item of kNearest) {
    counts[item.label] = (counts[item.label] || 0) + 1;
  }

  // Return most common label
  let maxLabel = null;
  let maxCount = 0;
  for (const [label, count] of Object.entries(counts)) {
    if (count > maxCount) {
      maxCount = count;
      maxLabel = label;
    }
  }

  return maxLabel;
}`,
    testCases: [
      { id: '1', description: 'Should classify point near A as A', input: "knn([{point:[0,0],label:'A'},{point:[1,1],label:'A'},{point:[5,5],label:'B'},{point:[6,6],label:'B'}], [0.5, 0.5], 3)", expectedOutput: '"A"' },
      { id: '2', description: 'Should classify point near B as B', input: "knn([{point:[0,0],label:'A'},{point:[1,1],label:'A'},{point:[5,5],label:'B'},{point:[6,6],label:'B'}], [5.5, 5.5], 3)", expectedOutput: '"B"' },
      { id: '3', description: 'k=1 should return nearest neighbor label', input: "knn([{point:[0,0],label:'X'},{point:[10,10],label:'Y'}], [1, 1], 1)", expectedOutput: '"X"' },
    ],
    tags: ['classification', 'knn', 'algorithms'],
  },
];

export function getExerciseById(id: number): PracticeExercise | undefined {
  return practiceExercises.find(ex => ex.id === id);
}

export function getExercisesByCategory(category: string): PracticeExercise[] {
  return practiceExercises.filter(ex => ex.category === category);
}

export function getExercisesByDifficulty(difficulty: 'beginner' | 'intermediate' | 'advanced'): PracticeExercise[] {
  return practiceExercises.filter(ex => ex.difficulty === difficulty);
}

export function getAllCategories(): string[] {
  return [...new Set(practiceExercises.map(ex => ex.category))];
}
