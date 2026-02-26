'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { learningPaths } from '@/data/learning-paths';
import { useAuth } from '@/context/AuthContext';
import { createClient } from '@/lib/supabase/client';
import Certificate from '@/components/Certificate';
import Logo from '@/components/Logo';

interface LessonProgress {
  lesson_id: number;
  completed: boolean;
}

interface PracticeProgressMap {
  [exerciseId: number]: { completed: boolean };
}

export default function CertificatePage() {
  const router = useRouter();
  const { user, loading: authLoading, signOut } = useAuth();
  const supabase = createClient();

  const [darkMode, setDarkMode] = useState(false);
  const [lessonProgress, setLessonProgress] = useState<LessonProgress[]>([]);
  const [practiceProgress, setPracticeProgress] = useState<PracticeProgressMap>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    setDarkMode(isDark);
    if (isDark) document.documentElement.classList.add('dark');
  }, []);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/auth/login?redirect=/certificate');
      return;
    }

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
      setLoading(false);
    };

    if (user) loadProgress();
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

  const isLessonCompleted = (id: number) => lessonProgress.some(p => p.lesson_id === id && p.completed);
  const isExerciseCompleted = (id: number) => practiceProgress[id]?.completed || false;

  const getPathCompletion = (path: typeof learningPaths[0]) => {
    const completedLessons = path.lessonIds.filter(id => isLessonCompleted(id)).length;
    const completedExercises = path.exerciseIds.filter(id => isExerciseCompleted(id)).length;
    const total = path.lessonIds.length + path.exerciseIds.length;
    const completed = completedLessons + completedExercises;
    return { completedLessons, completedExercises, total, completed, isComplete: completed === total && total > 0 };
  };

  const generateCertId = (pathId: string) => {
    if (!user) return '';
    const raw = `${user.id}-${pathId}`;
    let hash = 0;
    for (let i = 0; i < raw.length; i++) {
      const char = raw.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash |= 0;
    }
    return `SA-${Math.abs(hash).toString(36).toUpperCase().padStart(8, '0')}`;
  };

  if (authLoading || loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) return null;

  const completedPaths = learningPaths.filter(p => getPathCompletion(p).isComplete);
  const inProgressPaths = learningPaths.filter(p => !getPathCompletion(p).isComplete);

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
              <Link href="/paths" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Paths</Link>
              <Link href="/dashboard" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Dashboard</Link>
              <button onClick={toggleDarkMode} className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" aria-label="Toggle dark mode">
                {darkMode ? (
                  <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" /></svg>
                ) : (
                  <svg className="w-5 h-5 text-gray-600" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" /></svg>
                )}
              </button>
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-600 dark:text-gray-300 hidden sm:inline">{user.email}</span>
                <button onClick={() => signOut()} className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition">Sign out</button>
              </div>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Your Certificates</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">Complete all lessons and exercises in a learning path to earn your certificate.</p>

        {/* Earned Certificates */}
        {completedPaths.length > 0 && (
          <section className="mb-12">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Earned</h2>
            <div className="space-y-4">
              {completedPaths.map((path) => {
                const completion = getPathCompletion(path);
                return (
                  <div
                    key={path.id}
                    className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className="text-3xl">{path.icon}</span>
                        <div>
                          <h3 className="font-semibold text-gray-900 dark:text-white">{path.title}</h3>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {completion.completedLessons} lessons, {completion.completedExercises} exercises
                          </p>
                        </div>
                      </div>
                      <Certificate
                        pathTitle={path.title}
                        pathIcon={path.icon}
                        userEmail={user.email || ''}
                        completionDate={new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                        lessonsCompleted={completion.completedLessons}
                        exercisesCompleted={completion.completedExercises}
                        certificateId={generateCertId(path.id)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* In Progress */}
        <section>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            {completedPaths.length > 0 ? 'In Progress' : 'Learning Paths'}
          </h2>
          <div className="space-y-4">
            {inProgressPaths.map((path) => {
              const completion = getPathCompletion(path);
              const percent = Math.round((completion.completed / completion.total) * 100);

              return (
                <Link
                  key={path.id}
                  href={`/paths/${path.id}`}
                  className="block bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 transition-all"
                >
                  <div className="flex items-center gap-4 mb-3">
                    <span className="text-3xl">{path.icon}</span>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 dark:text-white">{path.title}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {completion.completed}/{completion.total} completed
                      </p>
                    </div>
                    <span className="text-sm font-bold text-blue-600 dark:text-blue-400">{percent}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${path.color} transition-all`}
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
