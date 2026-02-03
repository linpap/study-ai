export interface TestCase {
  id: string;
  description: string;
  input: string;
  expectedOutput: string;
  isHidden?: boolean;
}

export interface PracticeExercise {
  id: number;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  estimatedTime: string;
  problemStatement: string;
  hints: string[];
  language: 'javascript' | 'typescript' | 'python';
  starterCode: string;
  solutionCode: string;
  testCases: TestCase[];
  tags: string[];
}

export interface TestResult {
  testId: string;
  passed: boolean;
  input: string;
  expected: string;
  actual: string;
  error?: string;
}

export interface ExerciseProgress {
  completed: boolean;
  lastAttempt?: string;
  bestScore?: number;
  attempts: number;
}

export interface PracticeProgress {
  [exerciseId: number]: ExerciseProgress;
}
