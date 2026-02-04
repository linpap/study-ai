'use client';

import { useState, useEffect } from 'react';
import { PracticeExercise as PracticeExerciseType, TestResult } from '@/types/practice';
import CodeEditor from './CodeEditor';
import CodeBlock from './CodeBlock';
import { useCodeRunner } from './CodeRunner';

interface PracticeExerciseProps {
  exercise: PracticeExerciseType;
  darkMode: boolean;
  onComplete: (exerciseId: number, passed: boolean) => void;
}

export default function PracticeExercise({ exercise, darkMode, onComplete }: PracticeExerciseProps) {
  const [code, setCode] = useState(exercise.starterCode);
  const [results, setResults] = useState<TestResult[]>([]);
  const [running, setRunning] = useState(false);
  const [revealedHints, setRevealedHints] = useState(0);
  const [showSolution, setShowSolution] = useState(false);
  const [activeTab, setActiveTab] = useState<'problem' | 'solution'>('problem');
  const { runCode } = useCodeRunner();

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional reset when exercise changes
    setCode(exercise.starterCode);
    setResults([]);
    setRevealedHints(0);
    setShowSolution(false);
    setActiveTab('problem');
  }, [exercise.id, exercise.starterCode]);

  const handleRunTests = async () => {
    setRunning(true);
    const visibleTests = exercise.testCases.filter(t => !t.isHidden);
    const { results: testResults } = await runCode(code, visibleTests);
    setResults(testResults);
    setRunning(false);
  };

  const handleSubmit = async () => {
    setRunning(true);
    const { results: testResults } = await runCode(code, exercise.testCases);
    setResults(testResults.filter(r => !exercise.testCases.find(t => t.id === r.testId)?.isHidden));

    const allPassed = testResults.every(r => r.passed);
    onComplete(exercise.id, allPassed);
    setRunning(false);
  };

  const handleReset = () => {
    setCode(exercise.starterCode);
    setResults([]);
  };

  const revealNextHint = () => {
    if (revealedHints < exercise.hints.length) {
      setRevealedHints(prev => prev + 1);
    }
  };

  const passedCount = results.filter(r => r.passed).length;
  const totalTests = results.length;
  const allPassed = totalTests > 0 && passedCount === totalTests;

  const difficultyColors = {
    beginner: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    intermediate: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
    advanced: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
  };

  return (
    <div className="practice-exercise">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <span className={`px-2 py-1 text-xs font-medium rounded ${difficultyColors[exercise.difficulty]}`}>
            {exercise.difficulty.charAt(0).toUpperCase() + exercise.difficulty.slice(1)}
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {exercise.category}
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            ~{exercise.estimatedTime}
          </span>
        </div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          {exercise.title}
        </h1>
        <p className="text-gray-600 dark:text-gray-300 mt-1">
          {exercise.description}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Problem & Solution */}
        <div className="space-y-4">
          {/* Tabs */}
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setActiveTab('problem')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'problem'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Problem
            </button>
            <button
              onClick={() => { setActiveTab('solution'); setShowSolution(true); }}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'solution'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Solution
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'problem' ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="prose dark:prose-invert max-w-none">
                <div
                  className="problem-statement"
                  dangerouslySetInnerHTML={{
                    __html: exercise.problemStatement
                      .replace(/```javascript\n([\s\S]*?)```/g, '<pre class="code-example"><code>$1</code></pre>')
                      .replace(/```\n([\s\S]*?)```/g, '<pre class="code-example"><code>$1</code></pre>')
                      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
                      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                      .replace(/\n/g, '<br>')
                  }}
                />
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
              {showSolution ? (
                <CodeBlock
                  code={exercise.solutionCode}
                  language={exercise.language}
                  showLineNumbers={true}
                />
              ) : (
                <div className="p-8 text-center">
                  <p className="text-gray-500 dark:text-gray-400 mb-4">
                    Are you sure you want to see the solution?
                  </p>
                  <button
                    onClick={() => setShowSolution(true)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Reveal Solution
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Hints */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                Hints ({revealedHints}/{exercise.hints.length})
              </h3>
              {revealedHints < exercise.hints.length && (
                <button
                  onClick={revealNextHint}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Reveal next hint
                </button>
              )}
            </div>
            {revealedHints > 0 ? (
              <ul className="space-y-2">
                {exercise.hints.slice(0, revealedHints).map((hint, idx) => (
                  <li
                    key={idx}
                    className="text-sm text-gray-600 dark:text-gray-300 pl-4 border-l-2 border-yellow-400"
                  >
                    {hint}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Click &quot;Reveal next hint&quot; if you&apos;re stuck.
              </p>
            )}
          </div>
        </div>

        {/* Right: Editor & Results */}
        <div className="space-y-4">
          {/* Code Editor */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {exercise.language.charAt(0).toUpperCase() + exercise.language.slice(1)}
              </span>
              <button
                onClick={handleReset}
                className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Reset Code
              </button>
            </div>
            <CodeEditor
              code={code}
              onChange={setCode}
              language={exercise.language}
              height="300px"
              darkMode={darkMode}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleRunTests}
              disabled={running}
              className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {running ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Running...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Run Tests
                </>
              )}
            </button>
            <button
              onClick={handleSubmit}
              disabled={running}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Submit
            </button>
          </div>

          {/* Test Results */}
          {results.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className={`px-4 py-3 border-b ${
                allPassed
                  ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800'
              }`}>
                <div className="flex items-center gap-2">
                  {allPassed ? (
                    <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  )}
                  <span className={`font-medium ${
                    allPassed ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'
                  }`}>
                    {allPassed ? 'All tests passed!' : `${passedCount}/${totalTests} tests passed`}
                  </span>
                </div>
              </div>
              <div className="divide-y divide-gray-200 dark:divide-gray-700">
                {results.map((result, idx) => {
                  const testCase = exercise.testCases.find(t => t.id === result.testId);
                  return (
                    <div key={result.testId} className="p-4">
                      <div className="flex items-start gap-3">
                        {result.passed ? (
                          <svg className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 dark:text-white">
                            Test {idx + 1}: {testCase?.description || result.input}
                          </p>
                          {!result.passed && (
                            <div className="mt-2 text-sm space-y-1">
                              <p className="text-gray-600 dark:text-gray-400">
                                <span className="font-medium">Input:</span>{' '}
                                <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">{result.input}</code>
                              </p>
                              <p className="text-gray-600 dark:text-gray-400">
                                <span className="font-medium">Expected:</span>{' '}
                                <code className="bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-1 rounded">{result.expected}</code>
                              </p>
                              <p className="text-gray-600 dark:text-gray-400">
                                <span className="font-medium">Got:</span>{' '}
                                <code className="bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-1 rounded">
                                  {result.error || result.actual || 'undefined'}
                                </code>
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
