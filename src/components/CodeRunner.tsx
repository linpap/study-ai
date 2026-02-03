'use client';

import { useRef, useCallback } from 'react';
import { TestCase, TestResult } from '@/types/practice';

interface CodeRunnerProps {
  onResults: (results: TestResult[]) => void;
  onError: (error: string) => void;
  onRunning: (running: boolean) => void;
}

export default function CodeRunner({ onResults, onError, onRunning }: CodeRunnerProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const runCode = useCallback(async (code: string, testCases: TestCase[]) => {
    onRunning(true);
    onError('');

    const results: TestResult[] = [];

    for (const testCase of testCases) {
      try {
        const result = await executeInSandbox(code, testCase);
        results.push(result);
      } catch (err) {
        results.push({
          testId: testCase.id,
          passed: false,
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: '',
          error: err instanceof Error ? err.message : 'Unknown error',
        });
      }
    }

    onResults(results);
    onRunning(false);
  }, [onResults, onError, onRunning]);

  const executeInSandbox = (code: string, testCase: TestCase): Promise<TestResult> => {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        resolve({
          testId: testCase.id,
          passed: false,
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: '',
          error: 'Execution timed out (5 seconds)',
        });
      }, 5000);

      try {
        // Create a safe execution context
        const wrappedCode = `
          ${code}

          try {
            const __result__ = ${testCase.input};
            JSON.stringify(__result__);
          } catch (e) {
            'ERROR: ' + e.message;
          }
        `;

        // Use Function constructor for safer execution (still sandboxed by same-origin policy)
        const fn = new Function(wrappedCode);
        const rawResult = fn();

        clearTimeout(timeout);

        const actual = typeof rawResult === 'string' && rawResult.startsWith('ERROR:')
          ? rawResult
          : JSON.stringify(rawResult);

        // Normalize comparison (handle JSON stringification differences)
        const normalizedActual = actual.replace(/\s/g, '');
        const normalizedExpected = testCase.expectedOutput.replace(/\s/g, '');

        resolve({
          testId: testCase.id,
          passed: normalizedActual === normalizedExpected,
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: actual,
        });
      } catch (err) {
        clearTimeout(timeout);
        resolve({
          testId: testCase.id,
          passed: false,
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: '',
          error: err instanceof Error ? err.message : 'Execution error',
        });
      }
    });
  };

  // Expose runCode method via ref-like pattern
  const runnerRef = useRef({ runCode });
  runnerRef.current.runCode = runCode;

  return (
    <div style={{ display: 'none' }}>
      <iframe
        ref={iframeRef}
        sandbox="allow-scripts"
        title="Code Runner Sandbox"
      />
    </div>
  );
}

// Export a hook for using the code runner
export function useCodeRunner() {
  const runCode = async (
    code: string,
    testCases: TestCase[]
  ): Promise<{ results: TestResult[]; error?: string }> => {
    const results: TestResult[] = [];

    for (const testCase of testCases) {
      try {
        const result = await executeTest(code, testCase);
        results.push(result);
      } catch (err) {
        results.push({
          testId: testCase.id,
          passed: false,
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: '',
          error: err instanceof Error ? err.message : 'Unknown error',
        });
      }
    }

    return { results };
  };

  return { runCode };
}

async function executeTest(code: string, testCase: TestCase): Promise<TestResult> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      resolve({
        testId: testCase.id,
        passed: false,
        input: testCase.input,
        expected: testCase.expectedOutput,
        actual: '',
        error: 'Execution timed out (5 seconds)',
      });
    }, 5000);

    try {
      const wrappedCode = `
        ${code}

        try {
          const __result__ = ${testCase.input};
          JSON.stringify(__result__);
        } catch (e) {
          'ERROR: ' + e.message;
        }
      `;

      const fn = new Function(wrappedCode);
      const rawResult = fn();

      clearTimeout(timeout);

      const actual = typeof rawResult === 'string' && rawResult.startsWith('ERROR:')
        ? rawResult
        : JSON.stringify(rawResult);

      const normalizedActual = actual.replace(/\s/g, '');
      const normalizedExpected = testCase.expectedOutput.replace(/\s/g, '');

      resolve({
        testId: testCase.id,
        passed: normalizedActual === normalizedExpected,
        input: testCase.input,
        expected: testCase.expectedOutput,
        actual: actual,
      });
    } catch (err) {
      clearTimeout(timeout);
      resolve({
        testId: testCase.id,
        passed: false,
        input: testCase.input,
        expected: testCase.expectedOutput,
        actual: '',
        error: err instanceof Error ? err.message : 'Execution error',
      });
    }
  });
}
