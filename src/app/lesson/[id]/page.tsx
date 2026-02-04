'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { getLessonById, lessons } from '@/data/lessons';
import Quiz from '@/components/Quiz';
import { useAuth } from '@/context/AuthContext';
import { createClient } from '@/lib/supabase/client';

const FREE_LESSONS = [1, 2, 3];

export default function LessonPage() {
  const params = useParams();
  const router = useRouter();
  const lessonId = parseInt(params.id as string);
  const lesson = getLessonById(lessonId);
  const { user, loading: authLoading } = useAuth();
  const supabase = createClient();

  const [showQuiz, setShowQuiz] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [authChecked, setAuthChecked] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Process code blocks to add copy buttons
  useEffect(() => {
    const processCodeBlocks = () => {
      const codeBlocks = document.querySelectorAll('.prose .code-block:not([data-processed])');
      codeBlocks.forEach((block, index) => {
        block.setAttribute('data-processed', 'true');

        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        // Create header
        const header = document.createElement('div');
        header.className = 'code-block-header';

        // Detect language from content or use default
        const content = block.textContent || '';
        let language = 'code';
        if (content.includes('import ') || content.includes('def ') || content.includes('class ')) {
          language = 'python';
        } else if (content.includes('function') || content.includes('const ') || content.includes('let ')) {
          language = 'javascript';
        } else if (content.includes('SELECT') || content.includes('FROM') || content.includes('WHERE')) {
          language = 'sql';
        } else if (content.includes('{') && content.includes(':')) {
          language = 'json';
        }

        const langSpan = document.createElement('span');
        langSpan.className = 'code-block-lang';
        langSpan.textContent = language;

        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-block-copy';
        copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>`;

        copyBtn.onclick = async () => {
          try {
            await navigator.clipboard.writeText(content);
            copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg><span>Copied!</span>`;
            copyBtn.classList.add('copied');
            setTimeout(() => {
              copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>Copy</span>`;
              copyBtn.classList.remove('copied');
            }, 2000);
          } catch (err) {
            console.error('Failed to copy:', err);
          }
        };

        header.appendChild(langSpan);
        header.appendChild(copyBtn);

        // Wrap the block
        block.parentNode?.insertBefore(wrapper, block);
        wrapper.appendChild(header);
        wrapper.appendChild(block);
      });
    };

    // Run after content renders
    const timer = setTimeout(processCodeBlocks, 100);
    return () => clearTimeout(timer);
  }, [lesson, showQuiz]);

  const isFreeLesson = FREE_LESSONS.includes(lessonId);

  // Check auth for non-free lessons (client-side backup for middleware)
  useEffect(() => {
    if (!authLoading) {
      if (!isFreeLesson && !user) {
        router.push(`/auth/login?redirect=/lesson/${lessonId}`);
      } else {
        setAuthChecked(true);
      }
    }
  }, [authLoading, user, isFreeLesson, lessonId, router]);

  useEffect(() => {
    // Apply dark mode from localStorage
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
      document.documentElement.classList.add('dark');
    } else if (savedDarkMode === 'false') {
      document.documentElement.classList.remove('dark');
    }

    // Mark lesson as viewed
    const markViewed = async () => {
      if (!lesson) return;

      if (user && supabase) {
        // Save to Supabase for authenticated users
        await supabase.from('user_progress').upsert({
          user_id: user.id,
          lesson_id: lessonId,
          viewed: true,
          last_viewed: new Date().toISOString(),
        }, {
          onConflict: 'user_id,lesson_id',
        });
      } else {
        // Save to localStorage for non-authenticated users
        const progress = JSON.parse(localStorage.getItem('studyai-progress') || '{}');
        progress[lessonId] = {
          ...progress[lessonId],
          viewed: true,
          lastViewed: new Date().toISOString(),
        };
        localStorage.setItem('studyai-progress', JSON.stringify(progress));
      }
    };

    if (authChecked && lesson) {
      markViewed();
    }
  }, [lessonId, lesson, user, authChecked, supabase]);

  // Show loading while checking auth
  if (authLoading || (!authChecked && !isFreeLesson)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="flex flex-col items-center gap-4">
          <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-gray-500 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Lesson not found</h1>
          <Link href="/" className="text-blue-500 hover:underline">
            Return to lessons
          </Link>
        </div>
      </div>
    );
  }

  const handleQuizComplete = async (correct: number, total: number) => {
    setScore({ correct, total });
    setQuizCompleted(true);

    const scorePercent = Math.round((correct / total) * 100);
    const completedAt = new Date().toISOString();

    if (user && supabase) {
      // Save to Supabase for authenticated users
      await supabase.from('user_progress').upsert({
        user_id: user.id,
        lesson_id: lessonId,
        viewed: true,
        completed: true,
        score: scorePercent,
        completed_at: completedAt,
        last_viewed: completedAt,
      }, {
        onConflict: 'user_id,lesson_id',
      });
    } else {
      // Save to localStorage for non-authenticated users
      const progress = JSON.parse(localStorage.getItem('studyai-progress') || '{}');
      progress[lessonId] = {
        ...progress[lessonId],
        completed: true,
        score: scorePercent,
        completedAt,
      };
      localStorage.setItem('studyai-progress', JSON.stringify(progress));
    }
  };

  const currentIndex = lessons.findIndex(l => l.id === lessonId);
  const lessonNumber = currentIndex + 1;
  const nextLesson = currentIndex < lessons.length - 1 ? lessons[currentIndex + 1] : null;
  const prevLesson = currentIndex > 0 ? lessons[currentIndex - 1] : null;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm sticky top-0 z-10">
        <div className="max-w-[1400px] mx-auto px-8 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            All Lessons
          </Link>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Lesson {lessonNumber} of {lessons.length}
          </span>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-8 py-8">
        {/* Lesson Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              lesson.difficulty === 'Beginner'
                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                : lesson.difficulty === 'Intermediate'
                ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
                : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
            }`}>
              {lesson.difficulty}
            </span>
            <span className="text-gray-500 dark:text-gray-400 text-sm">{lesson.duration}</span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            {lesson.title}
          </h1>
          <p className="text-gray-600 dark:text-gray-300">{lesson.description}</p>
        </div>

        {!showQuiz ? (
          <>
            {/* Lesson Content */}
            <article
              className="prose prose-lg dark:prose-invert max-w-none mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm"
              dangerouslySetInnerHTML={{ __html: lesson.content }}
            />

            {/* Start Quiz Button */}
            <div className="text-center">
              <button
                onClick={() => setShowQuiz(true)}
                className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                Take the Quiz ({lesson.questions.length} questions)
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Quiz */}
            <Quiz
              questions={lesson.questions}
              onComplete={handleQuizComplete}
            />

            {/* Post-Quiz Actions */}
            {quizCompleted && (
              <div className="mt-8 text-center space-y-4">
                <p className="text-lg text-gray-700 dark:text-gray-200">
                  {score.correct >= score.total * 0.7
                    ? "Great job! You're ready for the next lesson."
                    : 'Consider reviewing this lesson before moving on.'}
                </p>
                <div className="flex justify-center gap-4">
                  <button
                    onClick={() => {
                      setShowQuiz(false);
                      setQuizCompleted(false);
                      window.scrollTo(0, 0);
                    }}
                    className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    Review Lesson
                  </button>
                  {nextLesson && (
                    <Link
                      href={`/lesson/${nextLesson.id}`}
                      className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                    >
                      Next Lesson: {nextLesson.title}
                    </Link>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {/* Navigation */}
        {!showQuiz && (
          <div className="flex justify-between mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
            {prevLesson ? (
              <Link
                href={`/lesson/${prevLesson.id}`}
                className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-blue-500"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <div>
                  <p className="text-sm text-gray-400">Previous</p>
                  <p className="font-medium">{prevLesson.title}</p>
                </div>
              </Link>
            ) : (
              <div />
            )}
            {nextLesson && (
              <Link
                href={`/lesson/${nextLesson.id}`}
                className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-blue-500 text-right"
              >
                <div>
                  <p className="text-sm text-gray-400">Next</p>
                  <p className="font-medium">{nextLesson.title}</p>
                </div>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            )}
          </div>
        )}
      </main>

      {/* Custom styles for lesson content */}
      <style jsx global>{`
        .prose h2 {
          color: #1f2937;
          border-bottom: 2px solid #e5e7eb;
          padding-bottom: 0.5rem;
          margin-top: 2rem;
        }
        .dark .prose h2 {
          color: #f3f4f6;
          border-bottom-color: #374151;
        }
        .prose h3 {
          color: #374151;
        }
        .dark .prose h3 {
          color: #d1d5db;
        }
        .prose h4 {
          color: #4b5563;
        }
        .dark .prose h4 {
          color: #9ca3af;
        }
        .prose .highlight {
          background: linear-gradient(135deg, #eff6ff, #f0fdf4);
          border-left: 4px solid #3b82f6;
          padding: 1rem 1.5rem;
          border-radius: 0.5rem;
          margin: 1.5rem 0;
        }
        .dark .prose .highlight {
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 197, 94, 0.1));
        }
        .prose .highlight p {
          margin: 0.5rem 0;
        }
        .prose .timeline {
          border-left: 3px solid #3b82f6;
          padding-left: 1.5rem;
          margin: 1rem 0;
        }
        .prose .timeline p {
          margin: 0.75rem 0;
          position: relative;
        }
        .prose .timeline p::before {
          content: '';
          position: absolute;
          left: -1.75rem;
          top: 0.5rem;
          width: 0.75rem;
          height: 0.75rem;
          background: #3b82f6;
          border-radius: 50%;
        }
        .prose .code-block-wrapper {
          margin: 1.5rem 0;
          border-radius: 0.75rem;
          overflow: hidden;
          background: #1e293b;
          border: 1px solid #334155;
        }
        .prose .code-block-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 1rem;
          background: #0f172a;
          border-bottom: 1px solid #334155;
        }
        .prose .code-block-lang {
          font-size: 0.75rem;
          color: #94a3b8;
          font-family: monospace;
          text-transform: lowercase;
        }
        .prose .code-block-copy {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.375rem 0.75rem;
          font-size: 0.75rem;
          color: #94a3b8;
          background: transparent;
          border: 1px solid #475569;
          border-radius: 0.375rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        .prose .code-block-copy:hover {
          background: #334155;
          color: #e2e8f0;
        }
        .prose .code-block-copy.copied {
          color: #4ade80;
          border-color: #4ade80;
        }
        .prose .code-block {
          background: #1e293b;
          color: #e2e8f0;
          padding: 1rem;
          overflow-x: auto;
          font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
          font-size: 0.875rem;
          white-space: pre-wrap;
          line-height: 1.6;
          margin: 0;
          border-radius: 0;
        }
        .prose .code-block-wrapper + .code-block-wrapper {
          margin-top: 1rem;
        }
        .prose table {
          width: 100%;
          border-collapse: collapse;
        }
        .prose th, .prose td {
          border: 1px solid #e5e7eb;
          padding: 0.75rem;
          text-align: left;
        }
        .dark .prose th, .dark .prose td {
          border-color: #374151;
        }
        .prose th {
          background: #f3f4f6;
        }
        .dark .prose th {
          background: #374151;
        }
      `}</style>
    </div>
  );
}
