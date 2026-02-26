'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { Article } from '@/data/articles';
import { lessons } from '@/data/lessons';

export default function BlogPostClient({ article }: { article: Article }) {
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const relatedLessonData = article.relatedLessons
    .map(id => lessons.find(l => l.id === id))
    .filter(Boolean);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            StudyAI
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-blue-600">Lessons</Link>
            <Link href="/practice" className="text-gray-600 dark:text-gray-300 hover:text-blue-600">Practice</Link>
            <Link href="/blog" className="text-blue-600 dark:text-blue-400 font-medium">Blog</Link>
          </nav>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        {/* Breadcrumb */}
        <nav className="text-sm text-gray-500 dark:text-gray-400 mb-6">
          <Link href="/" className="hover:text-blue-600">Home</Link>
          <span className="mx-2">/</span>
          <Link href="/blog" className="hover:text-blue-600">Blog</Link>
          <span className="mx-2">/</span>
          <span className="text-gray-900 dark:text-white">{article.title}</span>
        </nav>

        {/* Article Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4 text-sm">
            <span className="px-2 py-0.5 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-full font-medium">
              {article.category}
            </span>
            <span className="text-gray-500 dark:text-gray-400">{article.readTime}</span>
            <span className="text-gray-400 dark:text-gray-500">
              {new Date(article.date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
            {article.title}
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">{article.excerpt}</p>
          <div className="mt-4 flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <div>
              <p className="font-medium text-gray-900 dark:text-white text-sm">{article.author}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Partner & CTO, Greensolz</p>
            </div>
          </div>
        </div>

        {/* Article Content */}
        <article
          className="prose prose-lg dark:prose-invert max-w-none bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8"
          dangerouslySetInnerHTML={{ __html: article.content }}
        />

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-8">
          {article.tags.map(tag => (
            <span key={tag} className="text-sm text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded-full">
              {tag}
            </span>
          ))}
        </div>

        {/* Related Lessons */}
        {relatedLessonData.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Related Lessons</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Go deeper with our interactive lessons on this topic:
            </p>
            <div className="grid gap-3">
              {relatedLessonData.map((lesson) => lesson && (
                <Link
                  key={lesson.id}
                  href={`/lesson/${lesson.id}`}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition group"
                >
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="font-bold text-blue-600 dark:text-blue-400 text-sm">{lesson.id}</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                      {lesson.title}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{lesson.difficulty} &middot; {lesson.duration}</p>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* CTA */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-3">Want to Master This Topic?</h2>
          <p className="text-blue-100 mb-6">
            Our interactive course goes way beyond articles. Get hands-on with 31 lessons, 25 coding exercises, and AI-evaluated quizzes.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/lesson/1" className="px-8 py-3 bg-white text-blue-600 rounded-lg font-medium hover:bg-gray-100 transition">
              Start Learning Free
            </Link>
            <Link href="/premium" className="px-8 py-3 border-2 border-white text-white rounded-lg font-medium hover:bg-white/10 transition">
              View Premium
            </Link>
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/" className="hover:text-gray-900 dark:hover:text-white">Lessons</Link>
            <Link href="/practice" className="hover:text-gray-900 dark:hover:text-white">Practice</Link>
            <Link href="/blog" className="hover:text-gray-900 dark:hover:text-white">Blog</Link>
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
