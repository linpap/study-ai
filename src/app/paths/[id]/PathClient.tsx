'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getPathById } from '@/data/learning-paths';
import { getLessonById } from '@/data/lessons';
import { getExerciseById } from '@/data/practice-exercises';
import { useAuth } from '@/context/AuthContext';
import { createClient } from '@/lib/supabase/client';
import Logo from '@/components/Logo';

interface LessonProgress {
  lesson_id: number;
  completed: boolean;
}

interface PracticeProgressMap {
  [exerciseId: number]: {
    completed: boolean;
  };
}

export default function PathClient({ pathId }: { pathId: string }) {
  const path = getPathById(pathId);
  const { user, loading: authLoading, signOut } = useAuth();
  const supabase = createClient();

  const [darkMode, setDarkMode] = useState(false);
  const [lessonProgress, setLessonProgress] = useState<LessonProgress[]>([]);
  const [practiceProgress, setPracticeProgress] = useState<PracticeProgressMap>({});

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    setDarkMode(isDark);
    if (isDark) document.documentElement.classList.add('dark');
  }, []);

  useEffect(() => {
    const loadProgress = async () => {
      if (user && supabase) {
        const { data } = await supabase
          .from('user_progress')
          .select('lesson_id, completed')
          .eq('user_id', user.id);
        if (data) setLessonProgress(data as LessonProgress[]);
      }

      const savedPractice = localStorage.getItem('practiceProgress');
      if (savedPractice) setPracticeProgress(JSON.parse(savedPractice));
    };

    loadProgress();
  }, [user, supabase]);

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('theme', newDarkMode ? 'dark' : 'light');
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  if (!path) return null;

  const isLessonCompleted = (id: number) => lessonProgress.some(p => p.lesson_id === id && p.completed);
  const isExerciseCompleted = (id: number) => practiceProgress[id]?.completed || false;

  const completedLessons = path.lessonIds.filter(id => isLessonCompleted(id)).length;
  const completedExercises = path.exerciseIds.filter(id => isExerciseCompleted(id)).length;
  const totalItems = path.lessonIds.length + path.exerciseIds.length;
  const completedItems = completedLessons + completedExercises;
  const progressPercent = totalItems > 0 ? Math.round((completedItems / totalItems) * 100) : 0;

  const difficultyColors: Record<string, string> = {
    Beginner: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    Intermediate: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    Advanced: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Logo size="sm" showText={true} />
            <nav className="flex items-center gap-4">
              <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Lessons</Link>
              <Link href="/practice" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Practice</Link>
              <Link href="/paths" className="text-blue-600 dark:text-blue-400 font-medium">Paths</Link>
              {user && (
                <Link href="/dashboard" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Dashboard</Link>
              )}
              <button onClick={toggleDarkMode} className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" aria-label="Toggle dark mode">
                {darkMode ? (
                  <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" /></svg>
                ) : (
                  <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" /></svg>
                )}
              </button>
              {!authLoading && (
                user ? (
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 dark:text-gray-300 hidden sm:inline">{user.email}</span>
                    <button onClick={() => signOut()} className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition">Sign out</button>
                  </div>
                ) : (
                  <Link href="/auth/login" className="px-4 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg hover:from-blue-600 hover:to-purple-700 transition">Sign in</Link>
                )
              )}
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Back link */}
        <Link href="/paths" className="inline-flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 mb-8 transition-colors">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          All Paths
        </Link>

        {/* Path Header */}
        <div className="mb-10">
          <div className="flex items-center gap-4 mb-4">
            <span className="text-5xl">{path.icon}</span>
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{path.title}</h1>
                <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${difficultyColors[path.difficulty]}`}>
                  {path.difficulty}
                </span>
              </div>
              <p className="text-gray-600 dark:text-gray-400">{path.description}</p>
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-6 text-sm text-gray-500 dark:text-gray-400 mb-6">
            <span>{path.lessonIds.length} lessons</span>
            <span>{path.exerciseIds.length} exercises</span>
            <span>~{path.estimatedHours} hours</span>
          </div>

          {/* Progress Bar */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Progress</span>
              <span className="text-sm font-bold text-blue-600 dark:text-blue-400">{completedItems}/{totalItems} completed</span>
            </div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r ${path.color} transition-all duration-500`}
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        </div>

        {/* Lessons */}
        <section className="mb-12">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Lessons</h2>
          <div className="space-y-3">
            {path.lessonIds.map((lessonId, index) => {
              const lesson = getLessonById(lessonId);
              const completed = isLessonCompleted(lessonId);

              return (
                <Link
                  key={lessonId}
                  href={`/lesson/${lessonId}`}
                  className="group flex items-center gap-4 bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all"
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    completed
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                  }`}>
                    {completed ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="text-sm font-bold">{index + 1}</span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors truncate">
                      {lesson?.title || `Lesson ${lessonId}`}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {lesson?.duration} &middot; {lesson?.difficulty}
                    </p>
                  </div>
                  <svg className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              );
            })}
          </div>
        </section>

        {/* Exercises */}
        <section className="mb-12">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Practice Exercises</h2>
          <div className="space-y-3">
            {path.exerciseIds.map((exerciseId) => {
              const exercise = getExerciseById(exerciseId);
              const completed = isExerciseCompleted(exerciseId);

              return (
                <Link
                  key={exerciseId}
                  href={`/practice/${exerciseId}`}
                  className="group flex items-center gap-4 bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700 hover:border-orange-400 dark:hover:border-orange-500 transition-all"
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    completed
                      ? 'bg-green-500 text-white'
                      : 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400'
                  }`}>
                    {completed ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                      </svg>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 dark:text-white group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors truncate">
                      {exercise?.title || `Exercise ${exerciseId}`}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {exercise?.estimatedTime} &middot; {exercise?.category}
                    </p>
                  </div>
                  <svg className="w-5 h-5 text-gray-400 group-hover:text-orange-500 transition flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              );
            })}
          </div>
        </section>

        {/* Start CTA */}
        <div className="text-center">
          <Link
            href={`/lesson/${path.lessonIds[0]}`}
            className={`inline-flex items-center gap-2 px-8 py-3.5 bg-gradient-to-r ${path.color} text-white font-semibold rounded-xl hover:opacity-90 transition shadow-lg`}
          >
            {completedItems > 0 ? 'Continue Path' : 'Start Path'}
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/blog" className="hover:text-gray-900 dark:hover:text-white">Blog</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms of Service</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
