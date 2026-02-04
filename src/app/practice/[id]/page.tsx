'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getExerciseById, practiceExercises } from '@/data/practice-exercises';
import { PracticeProgress } from '@/types/practice';
import PracticeExercise from '@/components/PracticeExercise';
import { useAuth } from '@/context/AuthContext';

export default function ExercisePage() {
  const params = useParams();
  const exerciseId = parseInt(params.id as string, 10);
  const exercise = getExerciseById(exerciseId);

  const [darkMode, setDarkMode] = useState(false);
  const [progress, setProgress] = useState<PracticeProgress>({});
  const { user, loading: authLoading, signOut } = useAuth();

  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional sync with localStorage on mount
    setDarkMode(savedDarkMode);
    if (savedDarkMode) {
      document.documentElement.classList.add('dark');
    }

    const savedProgress = localStorage.getItem('practiceProgress');
    if (savedProgress) {
      setProgress(JSON.parse(savedProgress));
    }
  }, []);

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', String(newDarkMode));
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const handleComplete = (id: number, passed: boolean) => {
    const newProgress = {
      ...progress,
      [id]: {
        ...progress[id],
        completed: passed || progress[id]?.completed || false,
        lastAttempt: new Date().toISOString(),
        attempts: (progress[id]?.attempts || 0) + 1,
        bestScore: passed ? 100 : (progress[id]?.bestScore || 0),
      },
    };
    setProgress(newProgress);
    localStorage.setItem('practiceProgress', JSON.stringify(newProgress));
  };

  // Navigation
  const currentIndex = practiceExercises.findIndex(ex => ex.id === exerciseId);
  const prevExercise = currentIndex > 0 ? practiceExercises[currentIndex - 1] : null;
  const nextExercise = currentIndex < practiceExercises.length - 1 ? practiceExercises[currentIndex + 1] : null;

  if (!exercise) {
    return (
      <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${darkMode ? 'dark' : ''}`}>
        <div className="max-w-4xl mx-auto px-4 py-16 text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Exercise Not Found
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-8">
            The exercise you&apos;re looking for doesn&apos;t exist.
          </p>
          <Link
            href="/practice"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Practice
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors ${darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">S</span>
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  StudyAI
                </span>
              </Link>
              <span className="text-gray-300 dark:text-gray-600">/</span>
              <Link
                href="/practice"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Practice
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                aria-label="Toggle dark mode"
              >
                {darkMode ? (
                  <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                  </svg>
                )}
              </button>
              {!authLoading && (
                user ? (
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 dark:text-gray-300 hidden sm:inline">
                      {user.email}
                    </span>
                    <button
                      onClick={() => signOut()}
                      className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                    >
                      Sign out
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth/login"
                    className="px-4 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg hover:from-blue-600 hover:to-purple-700 transition"
                  >
                    Sign in
                  </Link>
                )
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <PracticeExercise
          exercise={exercise}
          darkMode={darkMode}
          onComplete={handleComplete}
        />

        {/* Navigation */}
        <div className="flex items-center justify-between mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          {prevExercise ? (
            <Link
              href={`/practice/${prevExercise.id}`}
              className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <div className="text-left">
                <div className="text-xs text-gray-500 dark:text-gray-500">Previous</div>
                <div className="font-medium">{prevExercise.title}</div>
              </div>
            </Link>
          ) : (
            <div />
          )}

          {nextExercise ? (
            <Link
              href={`/practice/${nextExercise.id}`}
              className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              <div className="text-right">
                <div className="text-xs text-gray-500 dark:text-gray-500">Next</div>
                <div className="font-medium">{nextExercise.title}</div>
              </div>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          ) : (
            <div />
          )}
        </div>

        {/* Footer */}
        <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
              <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
              <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
              <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy Policy</Link>
              <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms of Service</Link>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
