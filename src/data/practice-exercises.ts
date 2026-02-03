import { PracticeExercise } from '@/types/practice';

export const practiceExercises: PracticeExercise[] = [
  {
    id: 1,
    title: 'Sum of Two Numbers',
    description: 'Write a function that returns the sum of two numbers.',
    difficulty: 'beginner',
    category: 'JavaScript Basics',
    estimatedTime: '5 min',
    problemStatement: `Write a function called \`sum\` that takes two numbers as arguments and returns their sum.

**Example:**
\`\`\`javascript
sum(2, 3) // returns 5
sum(-1, 1) // returns 0
sum(0, 0) // returns 0
\`\`\``,
    hints: [
      'Use the + operator to add two numbers',
      'Make sure to return the result, not just calculate it',
      'The function should work with negative numbers too',
    ],
    language: 'javascript',
    starterCode: `function sum(a, b) {
  // Your code here
}`,
    solutionCode: `function sum(a, b) {
  return a + b;
}`,
    testCases: [
      { id: '1', description: 'sum(2, 3) should return 5', input: 'sum(2, 3)', expectedOutput: '5' },
      { id: '2', description: 'sum(-1, 1) should return 0', input: 'sum(-1, 1)', expectedOutput: '0' },
      { id: '3', description: 'sum(0, 0) should return 0', input: 'sum(0, 0)', expectedOutput: '0' },
      { id: '4', description: 'sum(100, 200) should return 300', input: 'sum(100, 200)', expectedOutput: '300', isHidden: true },
    ],
    tags: ['basics', 'functions', 'arithmetic'],
  },
  {
    id: 2,
    title: 'Reverse a String',
    description: 'Write a function that reverses a given string.',
    difficulty: 'beginner',
    category: 'JavaScript Basics',
    estimatedTime: '10 min',
    problemStatement: `Write a function called \`reverseString\` that takes a string and returns it reversed.

**Example:**
\`\`\`javascript
reverseString("hello") // returns "olleh"
reverseString("world") // returns "dlrow"
reverseString("a") // returns "a"
\`\`\``,
    hints: [
      'You can convert a string to an array using .split("")',
      'Arrays have a .reverse() method',
      'You can join an array back to a string using .join("")',
    ],
    language: 'javascript',
    starterCode: `function reverseString(str) {
  // Your code here
}`,
    solutionCode: `function reverseString(str) {
  return str.split('').reverse().join('');
}`,
    testCases: [
      { id: '1', description: 'reverseString("hello") should return "olleh"', input: 'reverseString("hello")', expectedOutput: '"olleh"' },
      { id: '2', description: 'reverseString("world") should return "dlrow"', input: 'reverseString("world")', expectedOutput: '"dlrow"' },
      { id: '3', description: 'reverseString("a") should return "a"', input: 'reverseString("a")', expectedOutput: '"a"' },
      { id: '4', description: 'reverseString("") should return ""', input: 'reverseString("")', expectedOutput: '""', isHidden: true },
    ],
    tags: ['strings', 'arrays', 'basics'],
  },
  {
    id: 3,
    title: 'Find Maximum in Array',
    description: 'Write a function that finds the maximum value in an array.',
    difficulty: 'beginner',
    category: 'Arrays & Objects',
    estimatedTime: '10 min',
    problemStatement: `Write a function called \`findMax\` that takes an array of numbers and returns the maximum value.

**Example:**
\`\`\`javascript
findMax([1, 5, 3, 9, 2]) // returns 9
findMax([-1, -5, -3]) // returns -1
findMax([42]) // returns 42
\`\`\``,
    hints: [
      'You can use Math.max() with the spread operator',
      'Alternatively, loop through the array keeping track of the max',
      'Consider edge cases like arrays with one element',
    ],
    language: 'javascript',
    starterCode: `function findMax(arr) {
  // Your code here
}`,
    solutionCode: `function findMax(arr) {
  return Math.max(...arr);
}`,
    testCases: [
      { id: '1', description: 'findMax([1, 5, 3, 9, 2]) should return 9', input: 'findMax([1, 5, 3, 9, 2])', expectedOutput: '9' },
      { id: '2', description: 'findMax([-1, -5, -3]) should return -1', input: 'findMax([-1, -5, -3])', expectedOutput: '-1' },
      { id: '3', description: 'findMax([42]) should return 42', input: 'findMax([42])', expectedOutput: '42' },
      { id: '4', description: 'findMax([0, 0, 0]) should return 0', input: 'findMax([0, 0, 0])', expectedOutput: '0', isHidden: true },
    ],
    tags: ['arrays', 'math', 'basics'],
  },
  {
    id: 4,
    title: 'Count Vowels',
    description: 'Write a function that counts the number of vowels in a string.',
    difficulty: 'beginner',
    category: 'JavaScript Basics',
    estimatedTime: '10 min',
    problemStatement: `Write a function called \`countVowels\` that takes a string and returns the count of vowels (a, e, i, o, u).

**Example:**
\`\`\`javascript
countVowels("hello") // returns 2
countVowels("AEIOU") // returns 5
countVowels("xyz") // returns 0
\`\`\``,
    hints: [
      'Convert the string to lowercase to handle both cases',
      'You can use a regular expression to match vowels',
      'Or loop through each character and check if it is a vowel',
    ],
    language: 'javascript',
    starterCode: `function countVowels(str) {
  // Your code here
}`,
    solutionCode: `function countVowels(str) {
  const matches = str.toLowerCase().match(/[aeiou]/g);
  return matches ? matches.length : 0;
}`,
    testCases: [
      { id: '1', description: 'countVowels("hello") should return 2', input: 'countVowels("hello")', expectedOutput: '2' },
      { id: '2', description: 'countVowels("AEIOU") should return 5', input: 'countVowels("AEIOU")', expectedOutput: '5' },
      { id: '3', description: 'countVowels("xyz") should return 0', input: 'countVowels("xyz")', expectedOutput: '0' },
      { id: '4', description: 'countVowels("Programming") should return 3', input: 'countVowels("Programming")', expectedOutput: '3', isHidden: true },
    ],
    tags: ['strings', 'regex', 'basics'],
  },
  {
    id: 5,
    title: 'FizzBuzz',
    description: 'Implement the classic FizzBuzz algorithm.',
    difficulty: 'beginner',
    category: 'Functions',
    estimatedTime: '15 min',
    problemStatement: `Write a function called \`fizzBuzz\` that takes a number n and returns an array of strings from 1 to n where:
- Numbers divisible by 3 are replaced with "Fizz"
- Numbers divisible by 5 are replaced with "Buzz"
- Numbers divisible by both 3 and 5 are replaced with "FizzBuzz"
- Other numbers remain as strings

**Example:**
\`\`\`javascript
fizzBuzz(5) // returns ["1", "2", "Fizz", "4", "Buzz"]
fizzBuzz(15) // returns ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
\`\`\``,
    hints: [
      'Use the modulo operator (%) to check divisibility',
      'Check divisibility by 15 (both 3 and 5) first',
      'Use a loop to build the result array',
    ],
    language: 'javascript',
    starterCode: `function fizzBuzz(n) {
  // Your code here
}`,
    solutionCode: `function fizzBuzz(n) {
  const result = [];
  for (let i = 1; i <= n; i++) {
    if (i % 15 === 0) {
      result.push("FizzBuzz");
    } else if (i % 3 === 0) {
      result.push("Fizz");
    } else if (i % 5 === 0) {
      result.push("Buzz");
    } else {
      result.push(String(i));
    }
  }
  return result;
}`,
    testCases: [
      { id: '1', description: 'fizzBuzz(5) should return correct array', input: 'JSON.stringify(fizzBuzz(5))', expectedOutput: '["1","2","Fizz","4","Buzz"]' },
      { id: '2', description: 'fizzBuzz(3) should return ["1","2","Fizz"]', input: 'JSON.stringify(fizzBuzz(3))', expectedOutput: '["1","2","Fizz"]' },
      { id: '3', description: 'fizzBuzz(1) should return ["1"]', input: 'JSON.stringify(fizzBuzz(1))', expectedOutput: '["1"]' },
      { id: '4', description: 'fizzBuzz(15) should end with FizzBuzz', input: 'fizzBuzz(15)[14]', expectedOutput: '"FizzBuzz"', isHidden: true },
    ],
    tags: ['loops', 'conditionals', 'classic'],
  },
  {
    id: 6,
    title: 'Palindrome Checker',
    description: 'Check if a given string is a palindrome.',
    difficulty: 'intermediate',
    category: 'Functions',
    estimatedTime: '15 min',
    problemStatement: `Write a function called \`isPalindrome\` that takes a string and returns true if it reads the same forwards and backwards (ignoring case and non-alphanumeric characters).

**Example:**
\`\`\`javascript
isPalindrome("racecar") // returns true
isPalindrome("hello") // returns false
isPalindrome("A man, a plan, a canal: Panama") // returns true
\`\`\``,
    hints: [
      'Remove non-alphanumeric characters using a regex',
      'Convert to lowercase for case-insensitive comparison',
      'Compare the string with its reverse',
    ],
    language: 'javascript',
    starterCode: `function isPalindrome(str) {
  // Your code here
}`,
    solutionCode: `function isPalindrome(str) {
  const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');
  return cleaned === cleaned.split('').reverse().join('');
}`,
    testCases: [
      { id: '1', description: 'isPalindrome("racecar") should return true', input: 'isPalindrome("racecar")', expectedOutput: 'true' },
      { id: '2', description: 'isPalindrome("hello") should return false', input: 'isPalindrome("hello")', expectedOutput: 'false' },
      { id: '3', description: 'isPalindrome("A man, a plan, a canal: Panama") should return true', input: 'isPalindrome("A man, a plan, a canal: Panama")', expectedOutput: 'true' },
      { id: '4', description: 'isPalindrome("Was it a car or a cat I saw?") should return true', input: 'isPalindrome("Was it a car or a cat I saw?")', expectedOutput: 'true', isHidden: true },
    ],
    tags: ['strings', 'algorithms', 'classic'],
  },
  {
    id: 7,
    title: 'Array Flatten',
    description: 'Flatten a nested array to a single level.',
    difficulty: 'intermediate',
    category: 'Arrays & Objects',
    estimatedTime: '20 min',
    problemStatement: `Write a function called \`flatten\` that takes a nested array and returns a flattened array.

**Example:**
\`\`\`javascript
flatten([1, [2, 3], [4, [5, 6]]]) // returns [1, 2, 3, 4, 5, 6]
flatten([[1, 2], [3, 4]]) // returns [1, 2, 3, 4]
flatten([1, 2, 3]) // returns [1, 2, 3]
\`\`\``,
    hints: [
      'You can use recursion to handle deeply nested arrays',
      'Check if each element is an array using Array.isArray()',
      'The modern way is to use arr.flat(Infinity)',
    ],
    language: 'javascript',
    starterCode: `function flatten(arr) {
  // Your code here
}`,
    solutionCode: `function flatten(arr) {
  return arr.reduce((flat, item) => {
    return flat.concat(Array.isArray(item) ? flatten(item) : item);
  }, []);
}`,
    testCases: [
      { id: '1', description: 'flatten([1, [2, 3], [4, [5, 6]]]) should return [1,2,3,4,5,6]', input: 'JSON.stringify(flatten([1, [2, 3], [4, [5, 6]]]))', expectedOutput: '[1,2,3,4,5,6]' },
      { id: '2', description: 'flatten([[1, 2], [3, 4]]) should return [1,2,3,4]', input: 'JSON.stringify(flatten([[1, 2], [3, 4]]))', expectedOutput: '[1,2,3,4]' },
      { id: '3', description: 'flatten([1, 2, 3]) should return [1,2,3]', input: 'JSON.stringify(flatten([1, 2, 3]))', expectedOutput: '[1,2,3]' },
      { id: '4', description: 'flatten([[[1]], [[2]], [[3]]]) should return [1,2,3]', input: 'JSON.stringify(flatten([[[1]], [[2]], [[3]]]))', expectedOutput: '[1,2,3]', isHidden: true },
    ],
    tags: ['arrays', 'recursion', 'algorithms'],
  },
  {
    id: 8,
    title: 'Object Deep Clone',
    description: 'Create a deep copy of an object.',
    difficulty: 'intermediate',
    category: 'Arrays & Objects',
    estimatedTime: '20 min',
    problemStatement: `Write a function called \`deepClone\` that creates a deep copy of an object (including nested objects and arrays).

**Example:**
\`\`\`javascript
const obj = { a: 1, b: { c: 2 } };
const clone = deepClone(obj);
clone.b.c = 3;
obj.b.c // still 2 (original unchanged)
\`\`\``,
    hints: [
      'Handle different types: objects, arrays, primitives',
      'Use recursion for nested structures',
      'Check for null since typeof null is "object"',
    ],
    language: 'javascript',
    starterCode: `function deepClone(obj) {
  // Your code here
}`,
    solutionCode: `function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  if (Array.isArray(obj)) {
    return obj.map(item => deepClone(item));
  }
  const cloned = {};
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  return cloned;
}`,
    testCases: [
      { id: '1', description: 'Should clone simple object', input: 'JSON.stringify(deepClone({a: 1, b: 2}))', expectedOutput: '{"a":1,"b":2}' },
      { id: '2', description: 'Should clone nested object', input: 'JSON.stringify(deepClone({a: {b: {c: 1}}}))', expectedOutput: '{"a":{"b":{"c":1}}}' },
      { id: '3', description: 'Should clone arrays', input: 'JSON.stringify(deepClone([1, [2, 3]]))', expectedOutput: '[1,[2,3]]' },
      { id: '4', description: 'Clone should be independent', input: '(() => { const o = {a:{b:1}}; const c = deepClone(o); c.a.b = 2; return o.a.b; })()', expectedOutput: '1', isHidden: true },
    ],
    tags: ['objects', 'recursion', 'algorithms'],
  },
  {
    id: 9,
    title: 'Debounce Function',
    description: 'Implement a debounce utility function.',
    difficulty: 'advanced',
    category: 'Functions',
    estimatedTime: '25 min',
    problemStatement: `Write a function called \`debounce\` that takes a function and a delay, and returns a debounced version. The debounced function delays invoking the original function until after the delay has elapsed since the last call.

**Example:**
\`\`\`javascript
const debouncedLog = debounce(console.log, 1000);
debouncedLog("Hello"); // Won't log immediately
debouncedLog("World"); // Cancels previous, starts new timer
// After 1000ms: logs "World"
\`\`\``,
    hints: [
      'Use setTimeout to delay the function call',
      'Store the timeout ID so you can cancel it',
      'Use clearTimeout to cancel the previous timer',
      'Consider using ...args to pass arguments through',
    ],
    language: 'javascript',
    starterCode: `function debounce(fn, delay) {
  // Your code here
}`,
    solutionCode: `function debounce(fn, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}`,
    testCases: [
      { id: '1', description: 'Should return a function', input: 'typeof debounce(() => {}, 100)', expectedOutput: '"function"' },
      { id: '2', description: 'Debounced function should delay execution', input: '(() => { let count = 0; const d = debounce(() => count++, 50); d(); return count; })()', expectedOutput: '0' },
      { id: '3', description: 'Should pass arguments through', input: '(() => { let result; const d = debounce((x) => result = x, 0); d(42); return new Promise(r => setTimeout(() => r(result), 10)); })()', expectedOutput: '42' },
    ],
    tags: ['functions', 'closures', 'async', 'utility'],
  },
  {
    id: 10,
    title: 'Binary Search',
    description: 'Implement binary search algorithm.',
    difficulty: 'advanced',
    category: 'AI Concepts',
    estimatedTime: '25 min',
    problemStatement: `Write a function called \`binarySearch\` that takes a sorted array and a target value, and returns the index of the target if found, or -1 if not found.

Binary search is fundamental to many AI algorithms, including decision trees and efficient data retrieval.

**Example:**
\`\`\`javascript
binarySearch([1, 2, 3, 4, 5], 3) // returns 2
binarySearch([1, 2, 3, 4, 5], 6) // returns -1
binarySearch([10, 20, 30], 10) // returns 0
\`\`\``,
    hints: [
      'Maintain left and right pointers',
      'Calculate mid point and compare with target',
      'Narrow the search space by half each iteration',
      'Time complexity should be O(log n)',
    ],
    language: 'javascript',
    starterCode: `function binarySearch(arr, target) {
  // Your code here
}`,
    solutionCode: `function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return -1;
}`,
    testCases: [
      { id: '1', description: 'binarySearch([1,2,3,4,5], 3) should return 2', input: 'binarySearch([1,2,3,4,5], 3)', expectedOutput: '2' },
      { id: '2', description: 'binarySearch([1,2,3,4,5], 6) should return -1', input: 'binarySearch([1,2,3,4,5], 6)', expectedOutput: '-1' },
      { id: '3', description: 'binarySearch([10,20,30], 10) should return 0', input: 'binarySearch([10,20,30], 10)', expectedOutput: '0' },
      { id: '4', description: 'binarySearch([1,2,3,4,5], 5) should return 4', input: 'binarySearch([1,2,3,4,5], 5)', expectedOutput: '4', isHidden: true },
    ],
    tags: ['algorithms', 'search', 'ai-fundamentals'],
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
