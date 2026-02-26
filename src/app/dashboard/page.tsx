'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { createClient } from '@/lib/supabase/client';
import { lessons } from '@/data/lessons';
import { practiceExercises } from '@/data/practice-exercises';
import Logo from '@/components/Logo';
import { learningPaths } from '@/data/learning-paths';

interface LessonProgress {
  lesson_id: number;
  viewed: boolean;
  completed: boolean;
  score: number | null;
  completed_at: string | null;
  last_viewed: string | null;
}

interface PracticeProgress {
  [exerciseId: number]: {
    completed: boolean;
    lastAttempt: string;
    attempts: number;
    bestScore: number;
  };
}

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading, isPremium, signOut } = useAuth();
  const supabase = createClient();

  const [darkMode, setDarkMode] = useState(false);
  const [lessonProgress, setLessonProgress] = useState<LessonProgress[]>([]);
  const [practiceProgress, setPracticeProgress] = useState<PracticeProgress>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional sync with localStorage on mount
    setDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/auth/login?redirect=/dashboard');
      return;
    }

    const loadProgress = async () => {
      if (user && supabase) {
        // Load lesson progress from Supabase
        const { data: lessonData } = await supabase
          .from('user_progress')
          .select('*')
          .eq('user_id', user.id);

        if (lessonData) {
          setLessonProgress(lessonData as LessonProgress[]);
        }
      }

      // Load practice progress from localStorage
      const savedPracticeProgress = localStorage.getItem('practiceProgress');
      if (savedPracticeProgress) {
        setPracticeProgress(JSON.parse(savedPracticeProgress));
      }

      setLoading(false);
    };

    if (user) {
      loadProgress();
    }
  }, [user, authLoading, supabase, router]);

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

  // Calculate stats
  const completedLessons = lessonProgress.filter(p => p.completed).length;
  const completedPractice = Object.values(practiceProgress).filter(p => p.completed).length;
  const totalAttempts = Object.values(practiceProgress).reduce((sum, p) => sum + (p.attempts || 0), 0);

  const averageLessonScore = lessonProgress.length > 0
    ? Math.round(lessonProgress.reduce((sum, p) => sum + (p.score || 0), 0) / lessonProgress.filter(p => p.score).length) || 0
    : 0;

  // Get recent activity (last 5 items)
  const recentActivity = [
    ...lessonProgress
      .filter(p => p.last_viewed || p.completed_at)
      .map(p => ({
        type: 'lesson' as const,
        id: p.lesson_id,
        title: lessons.find(l => l.id === p.lesson_id)?.title || `Lesson ${p.lesson_id}`,
        action: p.completed ? 'Completed' : 'Viewed',
        date: p.completed_at || p.last_viewed || '',
      })),
    ...Object.entries(practiceProgress)
      .filter(([, p]) => p.lastAttempt)
      .map(([id, p]) => ({
        type: 'practice' as const,
        id: parseInt(id),
        title: practiceExercises.find(e => e.id === parseInt(id))?.title || `Exercise ${id}`,
        action: p.completed ? 'Completed' : 'Attempted',
        date: p.lastAttempt,
      })),
  ]
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, 5);

  if (authLoading || loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors ${darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Logo size="sm" showText={true} />
            </div>

            <nav className="flex items-center gap-4">
              <Link
                href="/"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Lessons
              </Link>
              <Link
                href="/practice"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Practice
              </Link>
              <Link
                href="/paths"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Paths
              </Link>
              <Link
                href="/dashboard"
                className="text-blue-600 dark:text-blue-400 font-medium"
              >
                Dashboard
              </Link>
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
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome back!
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Track your learning progress and continue where you left off.
          </p>
          {isPremium && (
            <span className="inline-flex items-center mt-2 px-3 py-1 rounded-full text-sm font-medium bg-gradient-to-r from-blue-500 to-purple-600 text-white">
              Premium Member
            </span>
          )}
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{completedLessons}/{lessons.length}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Lessons Completed</p>
              </div>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{ width: `${(completedLessons / lessons.length) * 100}%` }}
              />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{completedPractice}/{practiceExercises.length}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Exercises Completed</p>
              </div>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-orange-500 transition-all"
                style={{ width: `${(completedPractice / practiceExercises.length) * 100}%` }}
              />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{averageLessonScore}%</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Avg Quiz Score</p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{totalAttempts}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">Total Attempts</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Recent Activity */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Recent Activity
            </h2>
            {recentActivity.length > 0 ? (
              <div className="space-y-4">
                {recentActivity.map((activity, index) => (
                  <Link
                    key={`${activity.type}-${activity.id}-${index}`}
                    href={activity.type === 'lesson' ? `/lesson/${activity.id}` : `/practice/${activity.id}`}
                    className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition"
                  >
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      activity.type === 'lesson'
                        ? 'bg-blue-100 dark:bg-blue-900/30'
                        : 'bg-orange-100 dark:bg-orange-900/30'
                    }`}>
                      {activity.type === 'lesson' ? (
                        <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 dark:text-white truncate">
                        {activity.title}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {activity.action} {new Date(activity.date).toLocaleDateString()}
                      </p>
                    </div>
                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                No activity yet. Start learning to see your progress here!
              </p>
            )}
          </div>

          {/* Quick Actions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Continue Learning
            </h2>
            <div className="space-y-4">
              {/* Next Lesson */}
              {(() => {
                const nextLesson = lessons.find(
                  l => !lessonProgress.find(p => p.lesson_id === l.id && p.completed)
                );
                if (!nextLesson) return null;
                return (
                  <Link
                    href={`/lesson/${nextLesson.id}`}
                    className="block p-4 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 transition"
                  >
                    <p className="text-sm opacity-90 mb-1">Next Lesson</p>
                    <p className="font-semibold">{nextLesson.title}</p>
                    <p className="text-sm opacity-75 mt-1">{nextLesson.duration}</p>
                  </Link>
                );
              })()}

              {/* Next Exercise */}
              {(() => {
                const nextExercise = practiceExercises.find(
                  e => !practiceProgress[e.id]?.completed
                );
                if (!nextExercise) return null;
                return (
                  <Link
                    href={`/practice/${nextExercise.id}`}
                    className="block p-4 rounded-lg bg-gradient-to-r from-orange-500 to-red-500 text-white hover:from-orange-600 hover:to-red-600 transition"
                  >
                    <p className="text-sm opacity-90 mb-1">Next Exercise</p>
                    <p className="font-semibold">{nextExercise.title}</p>
                    <p className="text-sm opacity-75 mt-1">{nextExercise.estimatedTime}</p>
                  </Link>
                );
              })()}

              {/* Links */}
              <div className="grid grid-cols-2 gap-4 pt-2">
                <Link
                  href="/"
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-400 transition text-center"
                >
                  <svg className="w-6 h-6 mx-auto mb-2 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">All Lessons</p>
                </Link>
                <Link
                  href="/practice"
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-orange-500 dark:hover:border-orange-400 transition text-center"
                >
                  <svg className="w-6 h-6 mx-auto mb-2 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                  </svg>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">All Exercises</p>
                </Link>
                <Link
                  href="/paths"
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-purple-500 dark:hover:border-purple-400 transition text-center"
                >
                  <svg className="w-6 h-6 mx-auto mb-2 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">Learning Paths</p>
                </Link>
                <Link
                  href="/certificate"
                  className="p-4 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-green-500 dark:hover:border-green-400 transition text-center"
                >
                  <svg className="w-6 h-6 mx-auto mb-2 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                  </svg>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">Certificates</p>
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Learning Paths Progress */}
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Learning Paths</h2>
            <Link href="/paths" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">View all</Link>
          </div>
          <div className="space-y-4">
            {learningPaths.map((path) => {
              const completedPathLessons = path.lessonIds.filter(id => lessonProgress.some(p => p.lesson_id === id && p.completed)).length;
              const completedPathExercises = path.exerciseIds.filter(id => practiceProgress[id]?.completed).length;
              const pathTotal = path.lessonIds.length + path.exerciseIds.length;
              const pathCompleted = completedPathLessons + completedPathExercises;
              const pathPercent = pathTotal > 0 ? Math.round((pathCompleted / pathTotal) * 100) : 0;

              return (
                <Link key={path.id} href={`/paths/${path.id}`} className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition">
                  <span className="text-2xl">{path.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-medium text-gray-900 dark:text-white text-sm truncate">{path.title}</p>
                      <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">{pathCompleted}/{pathTotal}</span>
                    </div>
                    <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div className={`h-full bg-gradient-to-r ${path.color} transition-all`} style={{ width: `${pathPercent}%` }} />
                    </div>
                  </div>
                  {pathPercent === 100 && (
                    <Link href="/certificate" className="text-xs text-green-600 dark:text-green-400 font-medium whitespace-nowrap">Certificate</Link>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-gray-200 dark:border-gray-700 mt-16 pt-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms of Service</Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
