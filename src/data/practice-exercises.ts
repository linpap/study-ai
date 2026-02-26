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
  // ============== NEW EXERCISES ==============
  {
    id: 11,
    title: 'Matrix Multiplication',
    description: 'Implement matrix multiplication - the core operation in neural networks.',
    difficulty: 'beginner',
    category: 'Linear Algebra',
    estimatedTime: '15 min',
    problemStatement: `**Matrix multiplication** is fundamental to neural networks - every layer performs matrix operations.

Write a function called \`matmul\` that multiplies two 2D matrices.

**Rules:**
- Matrix A (m×n) × Matrix B (n×p) = Result (m×p)
- Result[i][j] = sum of A[i][k] × B[k][j] for all k

**Example:**
\`\`\`javascript
matmul([[1,2],[3,4]], [[5,6],[7,8]])
// returns [[19,22],[43,50]]
// [1×5+2×7, 1×6+2×8] = [19, 22]
// [3×5+4×7, 3×6+4×8] = [43, 50]
\`\`\``,
    hints: [
      'Create a result matrix with dimensions rows(A) × cols(B)',
      'Use three nested loops: row of A, col of B, and the sum index',
      'Each element is a dot product of a row from A and column from B',
    ],
    language: 'javascript',
    starterCode: `function matmul(A, B) {
  // Your code here
  // Multiply matrix A by matrix B
}`,
    solutionCode: `function matmul(A, B) {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;

  const result = [];
  for (let i = 0; i < rowsA; i++) {
    result[i] = [];
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}`,
    testCases: [
      { id: '1', description: '2x2 matrices', input: 'JSON.stringify(matmul([[1,2],[3,4]], [[5,6],[7,8]]))', expectedOutput: '"[[19,22],[43,50]]"' },
      { id: '2', description: 'Identity matrix', input: 'JSON.stringify(matmul([[1,0],[0,1]], [[5,6],[7,8]]))', expectedOutput: '"[[5,6],[7,8]]"' },
      { id: '3', description: '2x3 times 3x2', input: 'JSON.stringify(matmul([[1,2,3],[4,5,6]], [[1],[2],[3]]))', expectedOutput: '"[[14],[32]]"' },
    ],
    tags: ['linear algebra', 'matrices', 'neural networks'],
  },
  {
    id: 12,
    title: 'Linear Regression Prediction',
    description: 'Implement the prediction step of linear regression.',
    difficulty: 'beginner',
    category: 'Machine Learning',
    estimatedTime: '10 min',
    problemStatement: `**Linear regression** predicts a continuous value using a linear equation.

Write a function called \`linearPredict\` that predicts values using weights and bias.

**Formula:** y = w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ + b

**Example:**
\`\`\`javascript
linearPredict([2, 3], [0.5, 1.5], 1)
// weights: [0.5, 1.5], bias: 1
// prediction: 2×0.5 + 3×1.5 + 1 = 1 + 4.5 + 1 = 6.5
\`\`\``,
    hints: [
      'Multiply each feature by its corresponding weight',
      'Sum all the products',
      'Add the bias at the end',
    ],
    language: 'javascript',
    starterCode: `function linearPredict(features, weights, bias) {
  // Your code here
  // Return weighted sum + bias
}`,
    solutionCode: `function linearPredict(features, weights, bias) {
  let sum = 0;
  for (let i = 0; i < features.length; i++) {
    sum += features[i] * weights[i];
  }
  return sum + bias;
}`,
    testCases: [
      { id: '1', description: 'Basic prediction', input: 'linearPredict([2, 3], [0.5, 1.5], 1)', expectedOutput: '6.5' },
      { id: '2', description: 'Zero bias', input: 'linearPredict([1, 2, 3], [1, 1, 1], 0)', expectedOutput: '6' },
      { id: '3', description: 'Single feature', input: 'linearPredict([5], [2], 3)', expectedOutput: '13' },
    ],
    tags: ['regression', 'prediction', 'machine learning'],
  },
  {
    id: 13,
    title: 'Logistic Regression (Sigmoid Prediction)',
    description: 'Combine linear regression with sigmoid for binary classification.',
    difficulty: 'intermediate',
    category: 'Machine Learning',
    estimatedTime: '12 min',
    problemStatement: `**Logistic regression** uses sigmoid to convert linear output to probability (0-1).

Write a function called \`logisticPredict\` that returns the probability of class 1.

**Formula:** P(y=1) = sigmoid(w·x + b) = 1 / (1 + e^(-(w·x + b)))

**Example:**
\`\`\`javascript
logisticPredict([2, 3], [0.5, 0.5], -2)
// linear: 2×0.5 + 3×0.5 + (-2) = 0.5
// sigmoid(0.5) ≈ 0.622
\`\`\``,
    hints: [
      'First calculate the linear combination (dot product + bias)',
      'Then apply sigmoid: 1 / (1 + Math.exp(-z))',
      'Result should be between 0 and 1',
    ],
    language: 'javascript',
    starterCode: `function logisticPredict(features, weights, bias) {
  // Your code here
  // 1. Calculate linear combination
  // 2. Apply sigmoid
}`,
    solutionCode: `function logisticPredict(features, weights, bias) {
  let z = 0;
  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }
  z += bias;
  return 1 / (1 + Math.exp(-z));
}`,
    testCases: [
      { id: '1', description: 'Returns ~0.62 for z=0.5', input: 'Math.round(logisticPredict([2, 3], [0.5, 0.5], -2) * 100) / 100', expectedOutput: '0.62' },
      { id: '2', description: 'Returns 0.5 for z=0', input: 'logisticPredict([1, 1], [1, -1], 0)', expectedOutput: '0.5' },
      { id: '3', description: 'Close to 1 for large positive z', input: 'Math.round(logisticPredict([5], [2], 0) * 1000) / 1000', expectedOutput: '1' },
    ],
    tags: ['classification', 'logistic regression', 'probability'],
  },
  {
    id: 14,
    title: 'Precision, Recall, and F1 Score',
    description: 'Calculate key classification metrics from predictions.',
    difficulty: 'intermediate',
    category: 'Model Evaluation',
    estimatedTime: '15 min',
    problemStatement: `Calculate **precision**, **recall**, and **F1 score** - crucial metrics for classification.

Write a function called \`classificationMetrics\` that returns all three metrics.

**Formulas:**
- Precision = TP / (TP + FP) — "Of predicted positives, how many are correct?"
- Recall = TP / (TP + FN) — "Of actual positives, how many did we find?"
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Example:**
\`\`\`javascript
// actual: [1,1,1,0,0], predicted: [1,1,0,0,1]
// TP=2, FP=1, FN=1
// Precision=2/3, Recall=2/3, F1=2/3
\`\`\``,
    hints: [
      'Count TP (both actual and predicted are 1)',
      'Count FP (predicted 1, actual 0)',
      'Count FN (predicted 0, actual 1)',
      'Handle division by zero - return 0 if denominator is 0',
    ],
    language: 'javascript',
    starterCode: `function classificationMetrics(actual, predicted) {
  // Your code here
  // Return { precision, recall, f1 }
}`,
    solutionCode: `function classificationMetrics(actual, predicted) {
  let tp = 0, fp = 0, fn = 0;

  for (let i = 0; i < actual.length; i++) {
    if (predicted[i] === 1 && actual[i] === 1) tp++;
    else if (predicted[i] === 1 && actual[i] === 0) fp++;
    else if (predicted[i] === 0 && actual[i] === 1) fn++;
  }

  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);

  return { precision, recall, f1 };
}`,
    testCases: [
      { id: '1', description: 'Basic metrics', input: 'JSON.stringify(classificationMetrics([1,1,1,0,0], [1,1,0,0,1]))', expectedOutput: '{"precision":0.6666666666666666,"recall":0.6666666666666666,"f1":0.6666666666666666}' },
      { id: '2', description: 'Perfect prediction', input: 'JSON.stringify(classificationMetrics([1,0,1,0], [1,0,1,0]))', expectedOutput: '{"precision":1,"recall":1,"f1":1}' },
      { id: '3', description: 'All wrong', input: 'classificationMetrics([1,1,1], [0,0,0]).recall', expectedOutput: '0' },
    ],
    tags: ['metrics', 'evaluation', 'classification'],
  },
  {
    id: 15,
    title: 'Confusion Matrix',
    description: 'Build a confusion matrix from predictions and actual labels.',
    difficulty: 'beginner',
    category: 'Model Evaluation',
    estimatedTime: '10 min',
    problemStatement: `A **confusion matrix** shows the breakdown of predictions vs actual values.

Write a function called \`confusionMatrix\` that returns a 2x2 matrix for binary classification.

**Structure:**
\`\`\`
              Predicted
              0    1
Actual  0   [TN,  FP]
        1   [FN,  TP]
\`\`\`

**Example:**
\`\`\`javascript
confusionMatrix([0,0,1,1], [0,1,0,1])
// returns [[1,1],[1,1]]
// TN=1, FP=1, FN=1, TP=1
\`\`\``,
    hints: [
      'Initialize a 2x2 matrix with zeros',
      'Loop through pairs of (actual, predicted)',
      'matrix[actual][predicted]++ for each pair',
    ],
    language: 'javascript',
    starterCode: `function confusionMatrix(actual, predicted) {
  // Your code here
  // Return [[TN, FP], [FN, TP]]
}`,
    solutionCode: `function confusionMatrix(actual, predicted) {
  const matrix = [[0, 0], [0, 0]];

  for (let i = 0; i < actual.length; i++) {
    matrix[actual[i]][predicted[i]]++;
  }

  return matrix;
}`,
    testCases: [
      { id: '1', description: 'Mixed results', input: 'JSON.stringify(confusionMatrix([0,0,1,1], [0,1,0,1]))', expectedOutput: '"[[1,1],[1,1]]"' },
      { id: '2', description: 'All correct', input: 'JSON.stringify(confusionMatrix([0,0,1,1], [0,0,1,1]))', expectedOutput: '"[[2,0],[0,2]]"' },
      { id: '3', description: 'All wrong', input: 'JSON.stringify(confusionMatrix([0,0,1,1], [1,1,0,0]))', expectedOutput: '"[[0,2],[2,0]]"' },
    ],
    tags: ['metrics', 'evaluation', 'visualization'],
  },
  {
    id: 16,
    title: 'Min-Max Normalization',
    description: 'Scale features to a 0-1 range using min-max normalization.',
    difficulty: 'beginner',
    category: 'Data Preprocessing',
    estimatedTime: '10 min',
    problemStatement: `**Min-Max normalization** scales data to [0, 1] range - essential for neural networks.

Write a function called \`minMaxNormalize\` that normalizes an array of values.

**Formula:** x_normalized = (x - min) / (max - min)

**Example:**
\`\`\`javascript
minMaxNormalize([10, 20, 30, 40, 50])
// min=10, max=50, range=40
// returns [0, 0.25, 0.5, 0.75, 1]
\`\`\``,
    hints: [
      'First find the min and max of the array',
      'Calculate range = max - min',
      'Apply formula to each element',
      'Handle edge case: if all values are the same, return array of 0s',
    ],
    language: 'javascript',
    starterCode: `function minMaxNormalize(arr) {
  // Your code here
  // Return normalized array
}`,
    solutionCode: `function minMaxNormalize(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min;

  if (range === 0) return arr.map(() => 0);

  return arr.map(x => (x - min) / range);
}`,
    testCases: [
      { id: '1', description: 'Basic normalization', input: 'JSON.stringify(minMaxNormalize([10, 20, 30, 40, 50]))', expectedOutput: '"[0,0.25,0.5,0.75,1]"' },
      { id: '2', description: 'Negative values', input: 'JSON.stringify(minMaxNormalize([-10, 0, 10]))', expectedOutput: '"[0,0.5,1]"' },
      { id: '3', description: 'All same values', input: 'JSON.stringify(minMaxNormalize([5, 5, 5]))', expectedOutput: '"[0,0,0]"' },
    ],
    tags: ['preprocessing', 'normalization', 'feature scaling'],
  },
  {
    id: 17,
    title: 'One-Hot Encoding',
    description: 'Convert categorical labels to one-hot vectors.',
    difficulty: 'beginner',
    category: 'Data Preprocessing',
    estimatedTime: '12 min',
    problemStatement: `**One-hot encoding** converts categories to binary vectors - required for neural networks.

Write a function called \`oneHotEncode\` that converts labels to one-hot vectors.

**Example:**
\`\`\`javascript
oneHotEncode(['cat', 'dog', 'cat', 'bird'], ['cat', 'dog', 'bird'])
// returns:
// [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
\`\`\``,
    hints: [
      'Create a vector of zeros with length = number of classes',
      'Set the index corresponding to the class to 1',
      'Use the classes array to find the index of each label',
    ],
    language: 'javascript',
    starterCode: `function oneHotEncode(labels, classes) {
  // Your code here
  // Return array of one-hot vectors
}`,
    solutionCode: `function oneHotEncode(labels, classes) {
  return labels.map(label => {
    const vector = new Array(classes.length).fill(0);
    const index = classes.indexOf(label);
    if (index !== -1) vector[index] = 1;
    return vector;
  });
}`,
    testCases: [
      { id: '1', description: 'Three classes', input: "JSON.stringify(oneHotEncode(['cat', 'dog', 'cat', 'bird'], ['cat', 'dog', 'bird']))", expectedOutput: '"[[1,0,0],[0,1,0],[1,0,0],[0,0,1]]"' },
      { id: '2', description: 'Numbers as classes', input: 'JSON.stringify(oneHotEncode([0, 1, 2, 1], [0, 1, 2]))', expectedOutput: '"[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]"' },
      { id: '3', description: 'Two classes', input: "JSON.stringify(oneHotEncode(['yes', 'no', 'yes'], ['yes', 'no']))", expectedOutput: '"[[1,0],[0,1],[1,0]]"' },
    ],
    tags: ['preprocessing', 'encoding', 'categorical'],
  },
  {
    id: 18,
    title: 'Cross-Entropy Loss',
    description: 'Calculate cross-entropy loss for classification.',
    difficulty: 'intermediate',
    category: 'Neural Networks',
    estimatedTime: '12 min',
    problemStatement: `**Cross-entropy loss** measures how wrong classification predictions are.

Write a function called \`crossEntropyLoss\` that calculates the average loss.

**Formula:** L = -1/n × Σ[y×log(p) + (1-y)×log(1-p)]

Where y is actual (0 or 1) and p is predicted probability.

**Example:**
\`\`\`javascript
crossEntropyLoss([1, 0, 1], [0.9, 0.1, 0.8])
// -1/3 × [1×log(0.9) + 1×log(0.9) + 1×log(0.8)]
// ≈ 0.146
\`\`\``,
    hints: [
      'Use Math.log for natural logarithm',
      'Clip probabilities to avoid log(0) - use small epsilon like 1e-15',
      'Sum the loss for each sample, then divide by n',
    ],
    language: 'javascript',
    starterCode: `function crossEntropyLoss(actual, predicted) {
  // Your code here
  // Return average cross-entropy loss
}`,
    solutionCode: `function crossEntropyLoss(actual, predicted) {
  const eps = 1e-15;
  let totalLoss = 0;

  for (let i = 0; i < actual.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predicted[i]));
    const y = actual[i];
    totalLoss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }

  return totalLoss / actual.length;
}`,
    testCases: [
      { id: '1', description: 'Good predictions', input: 'Math.round(crossEntropyLoss([1, 0, 1], [0.9, 0.1, 0.8]) * 1000) / 1000', expectedOutput: '0.146' },
      { id: '2', description: 'Perfect predictions', input: 'Math.round(crossEntropyLoss([1, 0], [0.999999, 0.000001]) * 1000) / 1000', expectedOutput: '0' },
      { id: '3', description: 'Bad predictions', input: 'crossEntropyLoss([1, 0], [0.1, 0.9]) > 2', expectedOutput: 'true' },
    ],
    tags: ['loss function', 'classification', 'neural networks'],
  },
  {
    id: 19,
    title: '2D Convolution',
    description: 'Apply a filter/kernel to an image using convolution.',
    difficulty: 'intermediate',
    category: 'Computer Vision',
    estimatedTime: '20 min',
    problemStatement: `**Convolution** is the core operation in CNNs - sliding a filter over an image.

Write a function called \`convolve2d\` that applies a 3x3 kernel to an image.

**Operation:** Slide the kernel, multiply element-wise, sum the result.

**Example:**
\`\`\`javascript
// 4x4 image, 3x3 edge detection kernel
// Output is 2x2 (no padding)
convolve2d(image, kernel)
\`\`\``,
    hints: [
      'Output size = (input_size - kernel_size + 1)',
      'For each output position, extract the corresponding patch from input',
      'Multiply patch with kernel element-wise and sum',
    ],
    language: 'javascript',
    starterCode: `function convolve2d(image, kernel) {
  // Your code here
  // Apply 3x3 kernel to image (no padding)
}`,
    solutionCode: `function convolve2d(image, kernel) {
  const imgH = image.length;
  const imgW = image[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;

  const outH = imgH - kH + 1;
  const outW = imgW - kW + 1;

  const output = [];

  for (let i = 0; i < outH; i++) {
    output[i] = [];
    for (let j = 0; j < outW; j++) {
      let sum = 0;
      for (let ki = 0; ki < kH; ki++) {
        for (let kj = 0; kj < kW; kj++) {
          sum += image[i + ki][j + kj] * kernel[ki][kj];
        }
      }
      output[i][j] = sum;
    }
  }

  return output;
}`,
    testCases: [
      { id: '1', description: 'Identity kernel', input: 'JSON.stringify(convolve2d([[1,2,3],[4,5,6],[7,8,9]], [[0,0,0],[0,1,0],[0,0,0]]))', expectedOutput: '"[[5]]"' },
      { id: '2', description: '4x4 image', input: 'convolve2d([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], [[1,1,1],[1,1,1],[1,1,1]])[0][0]', expectedOutput: '9' },
      { id: '3', description: 'Edge detection', input: 'convolve2d([[0,0,0,10],[0,0,0,10],[0,0,0,10]], [[-1,0,1],[-1,0,1],[-1,0,1]])[0][0]', expectedOutput: '30' },
    ],
    tags: ['cnn', 'convolution', 'computer vision'],
  },
  {
    id: 20,
    title: 'Max Pooling',
    description: 'Implement max pooling - a key operation in CNNs.',
    difficulty: 'intermediate',
    category: 'Computer Vision',
    estimatedTime: '15 min',
    problemStatement: `**Max pooling** reduces spatial dimensions by taking the max value in each region.

Write a function called \`maxPool2d\` with a 2x2 window and stride 2.

**Example:**
\`\`\`javascript
maxPool2d([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
// 2x2 regions: [1,2,5,6], [3,4,7,8], [9,10,13,14], [11,12,15,16]
// maxes: [[6,8],[14,16]]
\`\`\``,
    hints: [
      'Output size = input_size / pool_size',
      'For each output cell, find the corresponding 2x2 region in input',
      'Take the maximum value from that region',
    ],
    language: 'javascript',
    starterCode: `function maxPool2d(input) {
  // Your code here
  // 2x2 max pooling with stride 2
}`,
    solutionCode: `function maxPool2d(input) {
  const h = input.length;
  const w = input[0].length;
  const outH = Math.floor(h / 2);
  const outW = Math.floor(w / 2);

  const output = [];

  for (let i = 0; i < outH; i++) {
    output[i] = [];
    for (let j = 0; j < outW; j++) {
      const vals = [
        input[i*2][j*2],
        input[i*2][j*2+1],
        input[i*2+1][j*2],
        input[i*2+1][j*2+1]
      ];
      output[i][j] = Math.max(...vals);
    }
  }

  return output;
}`,
    testCases: [
      { id: '1', description: '4x4 to 2x2', input: 'JSON.stringify(maxPool2d([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))', expectedOutput: '"[[6,8],[14,16]]"' },
      { id: '2', description: '2x2 to 1x1', input: 'JSON.stringify(maxPool2d([[1,3],[2,4]]))', expectedOutput: '"[[4]]"' },
      { id: '3', description: 'Negative values', input: 'maxPool2d([[-5,-1],[-3,-2]])[0][0]', expectedOutput: '-1' },
    ],
    tags: ['cnn', 'pooling', 'computer vision'],
  },
  {
    id: 21,
    title: 'Simple Tokenizer',
    description: 'Build a basic tokenizer for text processing.',
    difficulty: 'beginner',
    category: 'NLP',
    estimatedTime: '10 min',
    problemStatement: `**Tokenization** splits text into tokens - the first step in NLP.

Write a function called \`tokenize\` that:
1. Converts to lowercase
2. Splits on whitespace and punctuation
3. Removes empty tokens

**Example:**
\`\`\`javascript
tokenize("Hello, World! How are you?")
// returns ["hello", "world", "how", "are", "you"]
\`\`\``,
    hints: [
      'Use toLowerCase() first',
      'Use a regex to split on non-word characters: /\\W+/',
      'Filter out empty strings',
    ],
    language: 'javascript',
    starterCode: `function tokenize(text) {
  // Your code here
  // Return array of tokens
}`,
    solutionCode: `function tokenize(text) {
  return text
    .toLowerCase()
    .split(/\\W+/)
    .filter(token => token.length > 0);
}`,
    testCases: [
      { id: '1', description: 'Basic sentence', input: 'JSON.stringify(tokenize("Hello, World!"))', expectedOutput: '["hello","world"]' },
      { id: '2', description: 'Multiple spaces', input: 'JSON.stringify(tokenize("one   two    three"))', expectedOutput: '["one","two","three"]' },
      { id: '3', description: 'Mixed punctuation', input: 'tokenize("AI is great... really!").length', expectedOutput: '4' },
    ],
    tags: ['nlp', 'tokenization', 'text processing'],
  },
  {
    id: 22,
    title: 'Bag of Words',
    description: 'Create a bag of words representation of text.',
    difficulty: 'intermediate',
    category: 'NLP',
    estimatedTime: '15 min',
    problemStatement: `**Bag of Words** represents text as word frequency counts.

Write a function called \`bagOfWords\` that counts word occurrences.

**Example:**
\`\`\`javascript
bagOfWords("the cat sat on the mat")
// returns { "the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1 }
\`\`\``,
    hints: [
      'First tokenize the text (lowercase, split on non-word chars)',
      'Use an object to count occurrences',
      'Increment count for each token',
    ],
    language: 'javascript',
    starterCode: `function bagOfWords(text) {
  // Your code here
  // Return object with word counts
}`,
    solutionCode: `function bagOfWords(text) {
  const tokens = text.toLowerCase().split(/\\W+/).filter(t => t.length > 0);
  const counts = {};

  for (const token of tokens) {
    counts[token] = (counts[token] || 0) + 1;
  }

  return counts;
}`,
    testCases: [
      { id: '1', description: 'Simple sentence', input: 'bagOfWords("the cat sat on the mat")["the"]', expectedOutput: '2' },
      { id: '2', description: 'Single words', input: 'Object.keys(bagOfWords("a b c d")).length', expectedOutput: '4' },
      { id: '3', description: 'Repeated word', input: 'bagOfWords("hello hello hello")["hello"]', expectedOutput: '3' },
    ],
    tags: ['nlp', 'bow', 'text representation'],
  },
  {
    id: 23,
    title: 'Train-Test Split',
    description: 'Split data into training and test sets.',
    difficulty: 'beginner',
    category: 'Data Preprocessing',
    estimatedTime: '10 min',
    problemStatement: `**Train-test split** divides data for model training and evaluation.

Write a function called \`trainTestSplit\` that splits data with a given ratio.

**Example:**
\`\`\`javascript
trainTestSplit([1,2,3,4,5,6,7,8,9,10], 0.8)
// returns { train: [1,2,3,4,5,6,7,8], test: [9,10] }
\`\`\``,
    hints: [
      'Calculate split index: Math.floor(data.length * ratio)',
      'Use slice to split the array',
      'Return an object with train and test arrays',
    ],
    language: 'javascript',
    starterCode: `function trainTestSplit(data, trainRatio) {
  // Your code here
  // Return { train: [...], test: [...] }
}`,
    solutionCode: `function trainTestSplit(data, trainRatio) {
  const splitIndex = Math.floor(data.length * trainRatio);
  return {
    train: data.slice(0, splitIndex),
    test: data.slice(splitIndex)
  };
}`,
    testCases: [
      { id: '1', description: '80-20 split', input: 'trainTestSplit([1,2,3,4,5,6,7,8,9,10], 0.8).train.length', expectedOutput: '8' },
      { id: '2', description: '50-50 split', input: 'trainTestSplit([1,2,3,4], 0.5).test.length', expectedOutput: '2' },
      { id: '3', description: 'Test set values', input: 'JSON.stringify(trainTestSplit([1,2,3,4,5], 0.6).test)', expectedOutput: '"[4,5]"' },
    ],
    tags: ['preprocessing', 'data splitting', 'evaluation'],
  },
  {
    id: 24,
    title: 'K-Means Clustering (One Step)',
    description: 'Implement one iteration of the K-Means algorithm.',
    difficulty: 'advanced',
    category: 'Machine Learning',
    estimatedTime: '20 min',
    problemStatement: `**K-Means** groups data points into k clusters.

Write a function called \`kmeansStep\` that performs one iteration:
1. Assign each point to nearest centroid
2. Update centroids to mean of assigned points

**Example:**
\`\`\`javascript
kmeansStep([[0,0],[1,1],[5,5],[6,6]], [[0,0],[5,5]])
// Assignments: [0,0,1,1]
// New centroids: [[0.5,0.5], [5.5,5.5]]
\`\`\``,
    hints: [
      'For each point, find the closest centroid using Euclidean distance',
      'Group points by their assigned centroid',
      'Calculate new centroid as mean of all assigned points',
    ],
    language: 'javascript',
    starterCode: `function kmeansStep(points, centroids) {
  // Your code here
  // Return { assignments: [...], newCentroids: [...] }
}`,
    solutionCode: `function kmeansStep(points, centroids) {
  // Assign points to nearest centroid
  const assignments = points.map(point => {
    let minDist = Infinity;
    let nearest = 0;

    centroids.forEach((centroid, idx) => {
      const dist = Math.sqrt(
        Math.pow(point[0] - centroid[0], 2) +
        Math.pow(point[1] - centroid[1], 2)
      );
      if (dist < minDist) {
        minDist = dist;
        nearest = idx;
      }
    });

    return nearest;
  });

  // Calculate new centroids
  const newCentroids = centroids.map((_, idx) => {
    const assigned = points.filter((_, i) => assignments[i] === idx);
    if (assigned.length === 0) return centroids[idx];

    const sumX = assigned.reduce((s, p) => s + p[0], 0);
    const sumY = assigned.reduce((s, p) => s + p[1], 0);
    return [sumX / assigned.length, sumY / assigned.length];
  });

  return { assignments, newCentroids };
}`,
    testCases: [
      { id: '1', description: 'Two clusters', input: 'JSON.stringify(kmeansStep([[0,0],[1,1],[5,5],[6,6]], [[0,0],[5,5]]).assignments)', expectedOutput: '"[0,0,1,1]"' },
      { id: '2', description: 'Updated centroids', input: 'kmeansStep([[0,0],[2,2]], [[0,0]]).newCentroids[0][0]', expectedOutput: '1' },
      { id: '3', description: 'Three points', input: 'kmeansStep([[0,0],[10,0],[5,5]], [[0,0],[10,10]]).assignments[2]', expectedOutput: '1' },
    ],
    tags: ['clustering', 'unsupervised', 'machine learning'],
  },
  {
    id: 25,
    title: 'Simple Attention Score',
    description: 'Calculate attention scores - the heart of Transformers.',
    difficulty: 'advanced',
    category: 'Deep Learning',
    estimatedTime: '15 min',
    problemStatement: `**Attention** lets models focus on relevant parts of input.

Write a function called \`attentionScores\` that computes attention weights.

**Steps:**
1. Compute dot products between query and all keys
2. Apply softmax to get weights that sum to 1

**Example:**
\`\`\`javascript
attentionScores([1,0], [[1,0],[0,1],[1,1]])
// dot products: [1, 0, 1]
// softmax: [0.422, 0.155, 0.422]
\`\`\``,
    hints: [
      'Compute dot product of query with each key',
      'Apply softmax: exp(x_i) / sum(exp(x_j))',
      'For numerical stability, subtract max before exp',
    ],
    language: 'javascript',
    starterCode: `function attentionScores(query, keys) {
  // Your code here
  // Return array of attention weights
}`,
    solutionCode: `function attentionScores(query, keys) {
  // Compute dot products
  const scores = keys.map(key => {
    let dot = 0;
    for (let i = 0; i < query.length; i++) {
      dot += query[i] * key[i];
    }
    return dot;
  });

  // Softmax with numerical stability
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);

  return expScores.map(e => e / sumExp);
}`,
    testCases: [
      { id: '1', description: 'Weights sum to 1', input: 'Math.round(attentionScores([1,0], [[1,0],[0,1]]).reduce((a,b)=>a+b,0) * 100) / 100', expectedOutput: '1' },
      { id: '2', description: 'Higher score = higher weight', input: 'attentionScores([1,0], [[1,0],[0,1]])[0] > attentionScores([1,0], [[1,0],[0,1]])[1]', expectedOutput: 'true' },
      { id: '3', description: 'Equal queries equal weights', input: 'Math.round(attentionScores([1,1], [[1,0],[0,1]])[0] * 100) / 100', expectedOutput: '0.5' },
    ],
    tags: ['attention', 'transformer', 'deep learning'],
  },
  {
    id: 26,
    title: 'Build a Simple RAG Pipeline',
    description: 'Implement a basic Retrieval-Augmented Generation pipeline with document chunking, similarity search, and answer generation.',
    difficulty: 'advanced',
    category: 'RAG',
    estimatedTime: '25 min',
    problemStatement: `**RAG (Retrieval-Augmented Generation)** grounds LLM responses in external knowledge by retrieving relevant documents before generating answers.

Write a function called \`ragPipeline\` that:
1. Chunks documents into sentences
2. Finds the most relevant chunk to a query using word overlap (Jaccard similarity)
3. Returns the best matching chunk as context

**Example:**
\`\`\`javascript
const docs = [
  "Neural networks learn by adjusting weights through backpropagation.",
  "RAG combines retrieval with generation for grounded AI responses.",
  "Transformers use self-attention to process sequences in parallel."
];
ragPipeline(docs, "How does RAG work?")
// returns "RAG combines retrieval with generation for grounded AI responses."
\`\`\``,
    hints: [
      'Tokenize both the query and each document into word sets (lowercase)',
      'Jaccard similarity = |intersection| / |union| of two word sets',
      'Return the document with the highest similarity score',
      'Handle edge cases: empty docs array should return empty string',
    ],
    language: 'javascript',
    starterCode: `function ragPipeline(documents, query) {
  // Your code here
  // 1. Tokenize query into a set of words
  // 2. For each document, compute Jaccard similarity with query
  // 3. Return the most relevant document
}`,
    solutionCode: `function ragPipeline(documents, query) {
  if (documents.length === 0) return "";

  const tokenize = (text) => new Set(text.toLowerCase().split(/\\W+/).filter(w => w.length > 0));
  const queryTokens = tokenize(query);

  let bestDoc = "";
  let bestScore = -1;

  for (const doc of documents) {
    const docTokens = tokenize(doc);
    const intersection = new Set([...queryTokens].filter(w => docTokens.has(w)));
    const union = new Set([...queryTokens, ...docTokens]);
    const score = union.size === 0 ? 0 : intersection.size / union.size;

    if (score > bestScore) {
      bestScore = score;
      bestDoc = doc;
    }
  }

  return bestDoc;
}`,
    testCases: [
      { id: '1', description: 'Finds RAG-related document', input: 'ragPipeline(["Neural networks use backpropagation.", "RAG combines retrieval with generation.", "CNNs process images."], "How does RAG work?")', expectedOutput: 'RAG combines retrieval with generation.' },
      { id: '2', description: 'Finds neural network document', input: 'ragPipeline(["The cat sat on the mat.", "Neural networks learn from data.", "Python is a language."], "How do neural networks learn?")', expectedOutput: 'Neural networks learn from data.' },
      { id: '3', description: 'Empty documents returns empty string', input: 'ragPipeline([], "any query")', expectedOutput: '' },
      { id: '4', description: 'Single document returns it', input: 'ragPipeline(["Only document here."], "document")', expectedOutput: 'Only document here.', isHidden: true },
    ],
    tags: ['rag', 'retrieval', 'nlp', 'similarity'],
  },
  {
    id: 27,
    title: 'Prompt Engineering: Few-Shot Classification',
    description: 'Build a few-shot classifier that uses example-based prompting to categorize text.',
    difficulty: 'intermediate',
    category: 'Prompt Engineering',
    estimatedTime: '15 min',
    problemStatement: `**Few-shot classification** uses labeled examples to classify new inputs without training a model.

Write a function called \`fewShotClassify\` that:
1. Takes an array of labeled examples \`{text, label}\`
2. Takes an input text to classify
3. Returns the label of the most similar example (using word overlap)

**Example:**
\`\`\`javascript
const examples = [
  { text: "I love this product, amazing!", label: "positive" },
  { text: "Terrible experience, worst ever", label: "negative" },
  { text: "It's okay, nothing special", label: "neutral" }
];
fewShotClassify(examples, "This is wonderful and great!")
// returns "positive"
\`\`\``,
    hints: [
      'Tokenize input and each example into word sets',
      'Calculate similarity between input and each example',
      'Return the label of the example with highest similarity',
      'Use Jaccard similarity: |intersection| / |union|',
    ],
    language: 'javascript',
    starterCode: `function fewShotClassify(examples, input) {
  // Your code here
  // Find the most similar example and return its label
}`,
    solutionCode: `function fewShotClassify(examples, input) {
  const tokenize = (text) => new Set(text.toLowerCase().split(/\\W+/).filter(w => w.length > 0));
  const inputTokens = tokenize(input);

  let bestLabel = examples[0].label;
  let bestScore = -1;

  for (const example of examples) {
    const exTokens = tokenize(example.text);
    const intersection = new Set([...inputTokens].filter(w => exTokens.has(w)));
    const union = new Set([...inputTokens, ...exTokens]);
    const score = union.size === 0 ? 0 : intersection.size / union.size;

    if (score > bestScore) {
      bestScore = score;
      bestLabel = example.label;
    }
  }

  return bestLabel;
}`,
    testCases: [
      { id: '1', description: 'Classifies positive sentiment', input: 'fewShotClassify([{text:"I love this",label:"positive"},{text:"I hate this",label:"negative"}], "I really love it")', expectedOutput: 'positive' },
      { id: '2', description: 'Classifies negative sentiment', input: 'fewShotClassify([{text:"Great product amazing",label:"positive"},{text:"Terrible awful horrible",label:"negative"}], "This is terrible and awful")', expectedOutput: 'negative' },
      { id: '3', description: 'Returns first label on tie', input: 'fewShotClassify([{text:"hello world",label:"greeting"},{text:"goodbye world",label:"farewell"}], "world")', expectedOutput: 'greeting' },
      { id: '4', description: 'Works with technical text', input: 'fewShotClassify([{text:"bug error crash",label:"bug"},{text:"new feature request",label:"feature"}], "application crashes with error")', expectedOutput: 'bug', isHidden: true },
    ],
    tags: ['prompt engineering', 'classification', 'few-shot', 'nlp'],
  },
  {
    id: 28,
    title: 'Sentiment Analysis',
    description: 'Build a rule-based sentiment analyzer using positive and negative word lists.',
    difficulty: 'intermediate',
    category: 'NLP',
    estimatedTime: '15 min',
    problemStatement: `**Sentiment analysis** determines whether text expresses positive, negative, or neutral sentiment.

Write a function called \`analyzeSentiment\` that:
1. Takes a text string
2. Counts positive words (good, great, love, amazing, excellent, happy, wonderful, fantastic, awesome, best)
3. Counts negative words (bad, terrible, hate, awful, worst, sad, horrible, poor, ugly, boring)
4. Returns an object with \`{score, label}\` where score = positive - negative counts, and label is "positive", "negative", or "neutral"

**Example:**
\`\`\`javascript
analyzeSentiment("This is a great and amazing product")
// { score: 2, label: "positive" }
\`\`\``,
    hints: [
      'Define arrays of positive and negative words',
      'Tokenize input to lowercase words',
      'Count how many tokens appear in each list',
      'Score > 0 = positive, < 0 = negative, 0 = neutral',
    ],
    language: 'javascript',
    starterCode: `function analyzeSentiment(text) {
  // Your code here
  // Return { score, label }
}`,
    solutionCode: `function analyzeSentiment(text) {
  const positive = new Set(["good", "great", "love", "amazing", "excellent", "happy", "wonderful", "fantastic", "awesome", "best"]);
  const negative = new Set(["bad", "terrible", "hate", "awful", "worst", "sad", "horrible", "poor", "ugly", "boring"]);

  const words = text.toLowerCase().split(/\\W+/).filter(w => w.length > 0);

  let posCount = 0;
  let negCount = 0;

  for (const word of words) {
    if (positive.has(word)) posCount++;
    if (negative.has(word)) negCount++;
  }

  const score = posCount - negCount;
  const label = score > 0 ? "positive" : score < 0 ? "negative" : "neutral";

  return { score, label };
}`,
    testCases: [
      { id: '1', description: 'Positive text', input: 'analyzeSentiment("This is great and amazing").label', expectedOutput: 'positive' },
      { id: '2', description: 'Negative text', input: 'analyzeSentiment("This is terrible and awful").label', expectedOutput: 'negative' },
      { id: '3', description: 'Neutral text', input: 'analyzeSentiment("This is a product").label', expectedOutput: 'neutral' },
      { id: '4', description: 'Score calculation', input: 'analyzeSentiment("good good bad").score', expectedOutput: '1', isHidden: true },
    ],
    tags: ['nlp', 'sentiment', 'text analysis'],
  },
  {
    id: 29,
    title: 'Named Entity Extraction',
    description: 'Extract named entities from text using pattern matching rules.',
    difficulty: 'intermediate',
    category: 'NLP',
    estimatedTime: '20 min',
    problemStatement: `**Named Entity Recognition (NER)** identifies and classifies entities like names, dates, and numbers in text.

Write a function called \`extractEntities\` that extracts:
- **NUMBERS**: any sequence of digits (optionally with decimals)
- **EMAILS**: patterns matching word@word.word
- **DATES**: patterns like MM/DD/YYYY or YYYY-MM-DD

Return an array of objects \`{type, value}\`.

**Example:**
\`\`\`javascript
extractEntities("Contact john@example.com by 12/25/2026 for $500")
// [
//   { type: "EMAIL", value: "john@example.com" },
//   { type: "DATE", value: "12/25/2026" },
//   { type: "NUMBER", value: "500" }
// ]
\`\`\``,
    hints: [
      'Use regex patterns for each entity type',
      'Email pattern: /[\\w.+-]+@[\\w-]+\\.[\\w.]+/g',
      'Date pattern: /\\d{1,2}\\/\\d{1,2}\\/\\d{4}|\\d{4}-\\d{2}-\\d{2}/g',
      'Number pattern: /\\b\\d+\\.?\\d*\\b/g — but extract after removing emails and dates to avoid conflicts',
    ],
    language: 'javascript',
    starterCode: `function extractEntities(text) {
  // Your code here
  // Return array of { type, value } objects
}`,
    solutionCode: `function extractEntities(text) {
  const entities = [];

  // Extract emails
  const emailRegex = /[\\w.+-]+@[\\w-]+\\.[\\w.]+/g;
  let match;
  while ((match = emailRegex.exec(text)) !== null) {
    entities.push({ type: "EMAIL", value: match[0] });
  }

  // Extract dates
  const dateRegex = /\\d{1,2}\\/\\d{1,2}\\/\\d{4}|\\d{4}-\\d{2}-\\d{2}/g;
  while ((match = dateRegex.exec(text)) !== null) {
    entities.push({ type: "DATE", value: match[0] });
  }

  // Extract numbers (exclude those already part of emails/dates)
  const cleaned = text.replace(/[\\w.+-]+@[\\w-]+\\.[\\w.]+/g, ' ').replace(/\\d{1,2}\\/\\d{1,2}\\/\\d{4}|\\d{4}-\\d{2}-\\d{2}/g, ' ');
  const numRegex = /\\b\\d+\\.?\\d*\\b/g;
  while ((match = numRegex.exec(cleaned)) !== null) {
    entities.push({ type: "NUMBER", value: match[0] });
  }

  return entities;
}`,
    testCases: [
      { id: '1', description: 'Extracts email', input: 'extractEntities("Email me at test@mail.com").find(e => e.type === "EMAIL").value', expectedOutput: 'test@mail.com' },
      { id: '2', description: 'Extracts date', input: 'extractEntities("Due on 01/15/2026").find(e => e.type === "DATE").value', expectedOutput: '01/15/2026' },
      { id: '3', description: 'Extracts number', input: 'extractEntities("Total is 42 items").find(e => e.type === "NUMBER").value', expectedOutput: '42' },
      { id: '4', description: 'Multiple entity types', input: 'extractEntities("Contact a@b.com on 2026-01-01 for 100").length', expectedOutput: '3', isHidden: true },
    ],
    tags: ['nlp', 'ner', 'regex', 'entity extraction'],
  },
  {
    id: 30,
    title: 'TF-IDF from Scratch',
    description: 'Implement Term Frequency-Inverse Document Frequency for text importance scoring.',
    difficulty: 'advanced',
    category: 'NLP',
    estimatedTime: '20 min',
    problemStatement: `**TF-IDF** measures how important a word is to a document in a collection.

Write a function called \`tfidf\` that:
1. Computes **TF** (term frequency) = count of term in doc / total words in doc
2. Computes **IDF** (inverse document frequency) = log(total docs / docs containing term)
3. Returns TF * IDF for a given term in a given document

**Example:**
\`\`\`javascript
const docs = ["the cat sat", "the dog ran", "the cat ran fast"];
tfidf(docs, 0, "cat")
// TF = 1/3 = 0.333
// IDF = log(3/2) = 0.405
// TF-IDF = 0.135
\`\`\``,
    hints: [
      'Tokenize each document into lowercase words',
      'TF = occurrences of term in doc / total words in doc',
      'IDF = Math.log(total number of docs / number of docs containing term)',
      'Handle edge case: if term appears in 0 docs, return 0',
    ],
    language: 'javascript',
    starterCode: `function tfidf(documents, docIndex, term) {
  // Your code here
  // Return the TF-IDF score
}`,
    solutionCode: `function tfidf(documents, docIndex, term) {
  const tokenize = (text) => text.toLowerCase().split(/\\W+/).filter(w => w.length > 0);
  const termLower = term.toLowerCase();

  // Tokenize target document
  const docTokens = tokenize(documents[docIndex]);
  const termCount = docTokens.filter(w => w === termLower).length;

  if (termCount === 0) return 0;

  // TF
  const tf = termCount / docTokens.length;

  // IDF
  const docsWithTerm = documents.filter(doc => tokenize(doc).includes(termLower)).length;
  const idf = Math.log(documents.length / docsWithTerm);

  return tf * idf;
}`,
    testCases: [
      { id: '1', description: 'TF-IDF for common word is low', input: 'Math.round(tfidf(["the cat sat", "the dog ran", "the cat ran"], 0, "the") * 1000) / 1000', expectedOutput: '0' },
      { id: '2', description: 'TF-IDF for unique word is higher', input: 'tfidf(["the cat sat", "the dog ran"], 0, "cat") > 0', expectedOutput: 'true' },
      { id: '3', description: 'Missing term returns 0', input: 'tfidf(["hello world", "foo bar"], 0, "xyz")', expectedOutput: '0' },
      { id: '4', description: 'Unique term has highest score', input: 'tfidf(["unique word here", "other words"], 0, "unique") > tfidf(["unique word here", "other words"], 0, "words")', expectedOutput: 'true', isHidden: true },
    ],
    tags: ['nlp', 'tf-idf', 'information retrieval', 'text mining'],
  },
  {
    id: 31,
    title: 'Q-Learning Grid World',
    description: 'Implement one step of Q-Learning update for a simple grid world agent.',
    difficulty: 'advanced',
    category: 'Reinforcement Learning',
    estimatedTime: '25 min',
    problemStatement: `**Q-Learning** is a model-free RL algorithm that learns action values for state-action pairs.

Write a function called \`qLearningStep\` that performs one Q-value update:

**Formula:** Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))

Parameters:
- \`qTable\`: object mapping "state-action" keys to Q-values
- \`state\`: current state (string)
- \`action\`: action taken (string)
- \`reward\`: reward received (number)
- \`nextState\`: resulting state (string)
- \`actions\`: array of possible actions
- \`alpha\`: learning rate (default 0.1)
- \`gamma\`: discount factor (default 0.9)

Return the updated Q-table.

**Example:**
\`\`\`javascript
const qTable = { "0-up": 0, "0-right": 0, "1-up": 5, "1-right": 3 };
qLearningStep(qTable, "0", "right", 1, "1", ["up", "right"], 0.1, 0.9)
// Q("0-right") = 0 + 0.1 * (1 + 0.9 * 5 - 0) = 0.55
\`\`\``,
    hints: [
      'Q-table key format: "state-action"',
      'Get current Q value: qTable[state + "-" + action] || 0',
      'Find max Q of next state across all actions',
      'Apply the update formula and return the modified table',
    ],
    language: 'javascript',
    starterCode: `function qLearningStep(qTable, state, action, reward, nextState, actions, alpha, gamma) {
  // Your code here
  // Update Q(state, action) and return updated qTable
}`,
    solutionCode: `function qLearningStep(qTable, state, action, reward, nextState, actions, alpha, gamma) {
  if (alpha === undefined) alpha = 0.1;
  if (gamma === undefined) gamma = 0.9;

  const key = state + "-" + action;
  const currentQ = qTable[key] || 0;

  // Find max Q-value for next state
  let maxNextQ = -Infinity;
  for (const a of actions) {
    const nextQ = qTable[nextState + "-" + a] || 0;
    if (nextQ > maxNextQ) maxNextQ = nextQ;
  }
  if (maxNextQ === -Infinity) maxNextQ = 0;

  // Update Q-value
  qTable[key] = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);

  return qTable;
}`,
    testCases: [
      { id: '1', description: 'Basic Q update', input: 'qLearningStep({"0-up":0,"0-right":0,"1-up":5,"1-right":3}, "0", "right", 1, "1", ["up","right"], 0.1, 0.9)["0-right"]', expectedOutput: '0.55' },
      { id: '2', description: 'Zero reward', input: 'qLearningStep({"s-a":0}, "s", "a", 0, "s2", ["a"], 0.1, 0.9)["s-a"]', expectedOutput: '0' },
      { id: '3', description: 'High reward updates significantly', input: 'qLearningStep({}, "s", "a", 10, "s2", ["a"], 1.0, 0)["s-a"]', expectedOutput: '10' },
      { id: '4', description: 'Preserves other Q values', input: 'qLearningStep({"x-y":5}, "a", "b", 1, "c", ["b"], 0.1, 0)["x-y"]', expectedOutput: '5', isHidden: true },
    ],
    tags: ['reinforcement learning', 'q-learning', 'agent', 'mdp'],
  },
  {
    id: 32,
    title: 'Build a Simple Chatbot',
    description: 'Create a rule-based chatbot with pattern matching and response templates.',
    difficulty: 'intermediate',
    category: 'LLM',
    estimatedTime: '20 min',
    problemStatement: `Build a simple **pattern-matching chatbot** that responds based on keyword detection.

Write a function called \`chatbot\` that:
1. Takes a user message string
2. Checks for keyword patterns (case-insensitive)
3. Returns the appropriate response

**Rules:**
- Contains "hello" or "hi" → "Hello! How can I help you today?"
- Contains "help" → "I can assist with AI, ML, and programming questions."
- Contains "bye" or "goodbye" → "Goodbye! Happy learning!"
- Contains "ai" or "artificial intelligence" → "AI is fascinating! Would you like to learn about machine learning or deep learning?"
- Contains "thanks" or "thank you" → "You're welcome! Let me know if you need anything else."
- Default → "I'm not sure I understand. Could you rephrase that?"

**Example:**
\`\`\`javascript
chatbot("Hello there!") // "Hello! How can I help you today?"
chatbot("Tell me about AI") // "AI is fascinating! ..."
\`\`\``,
    hints: [
      'Convert the input to lowercase for case-insensitive matching',
      'Use string.includes() to check for keywords',
      'Check patterns in order — first match wins',
      'Return the default response if no patterns match',
    ],
    language: 'javascript',
    starterCode: `function chatbot(message) {
  // Your code here
  // Match patterns and return appropriate response
}`,
    solutionCode: `function chatbot(message) {
  const msg = message.toLowerCase();

  if (msg.includes("hello") || msg.includes("hi")) {
    return "Hello! How can I help you today?";
  }
  if (msg.includes("help")) {
    return "I can assist with AI, ML, and programming questions.";
  }
  if (msg.includes("bye") || msg.includes("goodbye")) {
    return "Goodbye! Happy learning!";
  }
  if (msg.includes("ai") || msg.includes("artificial intelligence")) {
    return "AI is fascinating! Would you like to learn about machine learning or deep learning?";
  }
  if (msg.includes("thanks") || msg.includes("thank you")) {
    return "You're welcome! Let me know if you need anything else.";
  }

  return "I'm not sure I understand. Could you rephrase that?";
}`,
    testCases: [
      { id: '1', description: 'Greeting response', input: 'chatbot("Hello there!")', expectedOutput: 'Hello! How can I help you today?' },
      { id: '2', description: 'AI question', input: 'chatbot("Tell me about AI")', expectedOutput: 'AI is fascinating! Would you like to learn about machine learning or deep learning?' },
      { id: '3', description: 'Default response', input: 'chatbot("random gibberish xyz")', expectedOutput: "I'm not sure I understand. Could you rephrase that?" },
      { id: '4', description: 'Case insensitive', input: 'chatbot("GOODBYE")', expectedOutput: 'Goodbye! Happy learning!', isHidden: true },
    ],
    tags: ['chatbot', 'nlp', 'pattern matching', 'llm'],
  },
  {
    id: 33,
    title: 'Image Histogram Equalization',
    description: 'Implement histogram equalization to improve image contrast.',
    difficulty: 'intermediate',
    category: 'Computer Vision',
    estimatedTime: '15 min',
    problemStatement: `**Histogram equalization** enhances image contrast by redistributing pixel intensities.

Write a function called \`histogramEqualize\` that:
1. Takes a 1D array of pixel values (0-255)
2. Computes the histogram (frequency of each value)
3. Computes the cumulative distribution function (CDF)
4. Maps each pixel to its equalized value: round(CDF(pixel) * 255 / totalPixels)

**Example:**
\`\`\`javascript
histogramEqualize([0, 0, 1, 1, 2, 2, 3, 3])
// Histogram: {0:2, 1:2, 2:2, 3:2}
// CDF: {0:2, 1:4, 2:6, 3:8}
// Equalized: [64, 64, 127, 127, 191, 191, 255, 255]
\`\`\``,
    hints: [
      'Build a histogram: count occurrences of each pixel value (0-255)',
      'Compute CDF: cumulative sum of histogram values',
      'Normalize: equalized[i] = round(cdf[pixel[i]] * 255 / totalPixels)',
      'The minimum CDF value should map to 0 for proper equalization',
    ],
    language: 'javascript',
    starterCode: `function histogramEqualize(pixels) {
  // Your code here
  // Return array of equalized pixel values
}`,
    solutionCode: `function histogramEqualize(pixels) {
  const total = pixels.length;
  if (total === 0) return [];

  // Build histogram
  const histogram = new Array(256).fill(0);
  for (const p of pixels) {
    histogram[p]++;
  }

  // Compute CDF
  const cdf = new Array(256).fill(0);
  cdf[0] = histogram[0];
  for (let i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + histogram[i];
  }

  // Find minimum non-zero CDF
  const cdfMin = cdf.find(v => v > 0) || 0;

  // Map pixels
  return pixels.map(p => {
    return Math.round(((cdf[p] - cdfMin) / (total - cdfMin)) * 255);
  });
}`,
    testCases: [
      { id: '1', description: 'Uniform distribution', input: 'JSON.stringify(histogramEqualize([0, 0, 1, 1, 2, 2, 3, 3]))', expectedOutput: '"[0,0,73,73,146,146,255,255]"' },
      { id: '2', description: 'Single value unchanged', input: 'histogramEqualize([128, 128, 128])[0]', expectedOutput: '0' },
      { id: '3', description: 'Empty array', input: 'histogramEqualize([]).length', expectedOutput: '0' },
      { id: '4', description: 'Already spread values', input: 'histogramEqualize([0, 255]).length', expectedOutput: '2', isHidden: true },
    ],
    tags: ['computer vision', 'image processing', 'histogram', 'contrast'],
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
