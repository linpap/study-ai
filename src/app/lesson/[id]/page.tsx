'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { getLessonById, lessons } from '@/data/lessons';
import Quiz from '@/components/Quiz';

export default function LessonPage() {
  const params = useParams();
  const router = useRouter();
  const lessonId = parseInt(params.id as string);
  const lesson = getLessonById(lessonId);

  const [showQuiz, setShowQuiz] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [score, setScore] = useState({ correct: 0, total: 0 });

  useEffect(() => {
    // Apply dark mode from localStorage
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
      document.documentElement.classList.add('dark');
    } else if (savedDarkMode === 'false') {
      document.documentElement.classList.remove('dark');
    }

    // Mark lesson as viewed in localStorage
    if (lesson) {
      const progress = JSON.parse(localStorage.getItem('studyai-progress') || '{}');
      progress[lessonId] = {
        ...progress[lessonId],
        viewed: true,
        lastViewed: new Date().toISOString(),
      };
      localStorage.setItem('studyai-progress', JSON.stringify(progress));
    }
  }, [lessonId, lesson]);

  if (!lesson) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Lesson not found</h1>
          <Link href="/" className="text-blue-500 hover:underline">
            Return to lessons
          </Link>
        </div>
      </div>
    );
  }

  const handleQuizComplete = (correct: number, total: number) => {
    setScore({ correct, total });
    setQuizCompleted(true);

    // Save progress
    const progress = JSON.parse(localStorage.getItem('studyai-progress') || '{}');
    progress[lessonId] = {
      ...progress[lessonId],
      completed: true,
      score: Math.round((correct / total) * 100),
      completedAt: new Date().toISOString(),
    };
    localStorage.setItem('studyai-progress', JSON.stringify(progress));
  };

  const nextLesson = lessons.find(l => l.id === lessonId + 1);
  const prevLesson = lessons.find(l => l.id === lessonId - 1);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
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
            Lesson {lessonId} of {lessons.length}
          </span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
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
        .prose .code-block {
          background: #1f2937;
          color: #e5e7eb;
          padding: 1rem;
          border-radius: 0.5rem;
          overflow-x: auto;
          font-family: monospace;
          font-size: 0.875rem;
          white-space: pre-wrap;
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
