'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { lessons } from '@/data/lessons';
import { useAuth } from '@/context/AuthContext';
import { createClient } from '@/lib/supabase/client';
import Logo from '@/components/Logo';

const FREE_LESSONS = [1, 2, 3];

type LessonLockStatus = 'unlocked' | 'login_required' | 'premium_required';

interface Progress {
  [lessonId: number]: {
    viewed?: boolean;
    completed?: boolean;
    score?: number;
    completedAt?: string;
  };
}

interface UserProgress {
  lesson_id: number;
  viewed: boolean;
  completed: boolean;
  score: number | null;
  completed_at: string | null;
}

export default function Home() {
  const [progress, setProgress] = useState<Progress>({});
  const [darkMode, setDarkMode] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const { user, loading: authLoading, signOut, isPremium } = useAuth();
  const supabase = createClient();

  const getLockStatus = (lessonId: number): LessonLockStatus => {
    if (FREE_LESSONS.includes(lessonId)) return 'unlocked';
    if (!user) return 'login_required';
    if (!isPremium) return 'premium_required';
    return 'unlocked';
  };

  // Theme initialization - runs once on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    setDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    setMounted(true);
  }, []);

  // Progress loading - depends on auth state
  useEffect(() => {
    const loadProgress = async () => {
      if (user && supabase) {
        // Load from Supabase for authenticated users
        const { data } = await supabase
          .from('user_progress')
          .select('*')
          .eq('user_id', user.id);

        if (data) {
          const progressMap: Progress = {};
          (data as UserProgress[]).forEach((p) => {
            progressMap[p.lesson_id] = {
              viewed: p.viewed,
              completed: p.completed,
              score: p.score ?? undefined,
              completedAt: p.completed_at ?? undefined,
            };
          });
          setProgress(progressMap);
        }
      } else {
        // Load from localStorage for non-authenticated users
        const saved = localStorage.getItem('studyai-progress');
        if (saved) {
          setProgress(JSON.parse(saved));
        }
      }
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

  const completedCount = Object.values(progress).filter(p => p.completed).length;
  const overallProgress = Math.round((completedCount / lessons.length) * 100);

  // Create a sorted lessons array with proper sequential numbering
  const sortedLessons = [...lessons].map((lesson, index) => ({
    ...lesson,
    displayNumber: index + 1
  }));

  // Filter lessons based on search query
  const filteredLessons = sortedLessons.filter(lesson => {
    if (!searchQuery.trim()) return true;
    const query = searchQuery.toLowerCase();
    return (
      lesson.title.toLowerCase().includes(query) ||
      lesson.description.toLowerCase().includes(query) ||
      lesson.difficulty.toLowerCase().includes(query)
    );
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm sticky top-0 z-10 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <Logo size="md" showText={true} showTagline={true} />
          <nav className="flex items-center gap-4">
            <Link
              href="/"
              className="text-blue-600 dark:text-blue-400 font-medium"
            >
              Lessons
            </Link>
            <Link
              href="/practice"
              className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              Practice
            </Link>
            {user && (
              <Link
                href="/dashboard"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Dashboard
              </Link>
            )}
            {user && user.email === (process.env.NEXT_PUBLIC_ADMIN_EMAIL || 'linpap@gmail.com') && (
              <Link
                href="/admin"
                className="text-gray-600 dark:text-gray-300 hover:text-red-600 dark:hover:text-red-400 transition-colors"
              >
                Admin
              </Link>
            )}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
            >
              {!mounted ? (
                <div className="w-6 h-6" />
              ) : darkMode ? (
                <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-gray-500 dark:text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
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
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Learn <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">Artificial Intelligence</span>
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-2">
            Comprehensive lessons covering everything from basics to building your own AI applications. Interactive quizzes with voice input support.
          </p>
          <p className="text-sm text-gray-400 dark:text-gray-500 mb-4">by Greensolz</p>

          {/* Price Badge */}
          <div className="flex justify-center mb-8">
            <div className="price-badge inline-flex items-center gap-2 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800 rounded-full px-5 py-2.5 cursor-default">
              <span className="text-sm font-medium text-indigo-600 dark:text-indigo-300">Full Course</span>
              <span className="w-px h-4 bg-indigo-200 dark:bg-indigo-700"></span>
              <span className="price-badge-text text-lg font-bold">₹1,200</span>
            </div>
          </div>

          {/* Progress Overview */}
          <div className="inline-flex items-center gap-4 bg-white dark:bg-gray-800 rounded-2xl px-6 py-4 shadow-lg">
            <div className="text-left">
              <p className="text-sm text-gray-500 dark:text-gray-400">Your Progress</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {completedCount}/{lessons.length} Lessons
              </p>
            </div>
            <div className="w-24 h-24 relative">
              <svg className="w-24 h-24 transform -rotate-90">
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="8"
                  className="text-gray-200 dark:text-gray-700"
                />
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  fill="none"
                  stroke="url(#progress-gradient)"
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={`${overallProgress * 2.51} 251`}
                />
                <defs>
                  <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xl font-bold text-gray-900 dark:text-white">{overallProgress}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-4 gap-6 mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{lessons.length} Comprehensive Lessons</h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">From AI basics to building real applications</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Interactive Quizzes</h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">MCQ and descriptive questions with AI evaluation</p>
          </div>
          <Link href="/practice" className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm hover:shadow-md hover:border-orange-400 border-2 border-transparent transition-all group">
            <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">Practice Area</h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">Hands-on coding exercises with live execution</p>
          </Link>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
            <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Voice Input Support</h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm">Answer questions by speaking - great for learning</p>
          </div>
        </div>

        {/* Lessons Grid */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Course Content</h3>
          <div className="relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search lessons..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full sm:w-64 pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
        <div className="grid gap-4">
          {filteredLessons.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              No lessons found matching &quot;{searchQuery}&quot;
            </div>
          ) : null}
          {filteredLessons.map((lesson) => {
            const lessonProgress = progress[lesson.id];
            const isCompleted = lessonProgress?.completed;
            const isViewed = lessonProgress?.viewed;
            const lockStatus = getLockStatus(lesson.id);
            const isLocked = lockStatus !== 'unlocked';

            const getHref = () => {
              if (lockStatus === 'login_required') return '/auth/login?redirect=/lesson/' + lesson.id;
              if (lockStatus === 'premium_required') return '/premium?redirect=/lesson/' + lesson.id;
              return `/lesson/${lesson.id}`;
            };

            return (
              <Link
                key={lesson.id}
                href={getHref()}
                className={`group bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border-2 border-transparent ${
                  isLocked ? 'hover:border-purple-500' : 'hover:border-blue-500'
                }`}
              >
                <div className="flex items-start gap-4">
                  {/* Lesson Number / Status */}
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
                    isLocked
                      ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                      : isCompleted
                      ? 'bg-green-500 text-white'
                      : isViewed
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                  }`}>
                    {isLocked ? (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                    ) : isCompleted ? (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <span className="font-bold">{lesson.displayNumber}</span>
                    )}
                  </div>

                  {/* Lesson Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-1 flex-wrap">
                      <h4 className={`font-semibold text-gray-900 dark:text-white transition truncate ${
                        isLocked ? 'group-hover:text-purple-600 dark:group-hover:text-purple-400' : 'group-hover:text-blue-600 dark:group-hover:text-blue-400'
                      }`}>
                        {lesson.title}
                      </h4>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex-shrink-0 ${
                        lesson.difficulty === 'Beginner'
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : lesson.difficulty === 'Intermediate'
                          ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                          : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {lesson.difficulty}
                      </span>
                      {lockStatus === 'login_required' && (
                        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 flex-shrink-0">
                          Login required
                        </span>
                      )}
                      {lockStatus === 'premium_required' && (
                        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-gradient-to-r from-blue-100 to-purple-100 text-purple-700 dark:from-blue-900/30 dark:to-purple-900/30 dark:text-purple-400 flex-shrink-0">
                          Premium
                        </span>
                      )}
                    </div>
                    <p className="text-gray-600 dark:text-gray-300 text-sm mb-2 line-clamp-1">
                      {lesson.description}
                    </p>
                    <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                      <span className="flex items-center gap-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {lesson.duration}
                      </span>
                      <span className="flex items-center gap-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {lesson.questions.length} questions
                      </span>
                      {lessonProgress?.score !== undefined && (
                        <span className="flex items-center gap-1">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                          Score: {lessonProgress.score}%
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Arrow */}
                  <svg
                    className="w-6 h-6 text-gray-400 group-hover:text-blue-500 transition flex-shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400 mb-4">
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white transition">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white transition">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white transition">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white transition">Terms of Service</Link>
            <Link href="/refund" className="hover:text-gray-900 dark:hover:text-white transition">Refund Policy</Link>
          </div>
          <p className="text-center text-gray-500 dark:text-gray-400 text-sm">A product by Greensolz | Built for learning AI from the ground up</p>
        </footer>
      </main>
    </div>
  );
}
