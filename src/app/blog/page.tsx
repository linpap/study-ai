import { Metadata } from 'next';
import Link from 'next/link';
import { articles } from '@/data/articles';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';

export const metadata: Metadata = {
  title: 'Blog - AI & Machine Learning Articles',
  description: 'Learn AI and machine learning through in-depth articles, tutorials, and guides. From beginner concepts to advanced topics like transformers, RAG, and MLOps.',
  alternates: { canonical: '/blog' },
  openGraph: {
    title: 'StudyAI Blog',
    description: 'In-depth AI and machine learning articles, tutorials, and career guides.',
    url: `${siteConfig.url}/blog`,
  },
};

export default function BlogPage() {
  const categories = [...new Set(articles.map(a => a.category))];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            StudyAI
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400">Lessons</Link>
            <Link href="/practice" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400">Practice</Link>
            <Link href="/blog" className="text-blue-600 dark:text-blue-400 font-medium">Blog</Link>
          </nav>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        <JsonLd data={{
          '@context': 'https://schema.org',
          '@type': 'Blog',
          name: 'StudyAI Blog',
          description: 'AI and Machine Learning articles and tutorials',
          url: `${siteConfig.url}/blog`,
          publisher: {
            '@type': 'Organization',
            name: siteConfig.company,
          },
        }} />

        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Blog</h1>
        <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
          In-depth articles on AI, machine learning, and building your career in tech.
        </p>

        {/* Category Tags */}
        <div className="flex flex-wrap gap-2 mb-10">
          {categories.map(cat => (
            <span key={cat} className="px-3 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-full text-sm font-medium">
              {cat}
            </span>
          ))}
        </div>

        {/* Articles Grid */}
        <div className="grid gap-6">
          {articles.map((article) => (
            <Link
              key={article.slug}
              href={`/blog/${article.slug}`}
              className="group bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-all border-2 border-transparent hover:border-blue-500"
            >
              <div className="flex items-center gap-3 mb-3 text-sm">
                <span className="px-2 py-0.5 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-full font-medium">
                  {article.category}
                </span>
                <span className="text-gray-500 dark:text-gray-400">{article.readTime}</span>
                <span className="text-gray-400 dark:text-gray-500">
                  {new Date(article.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
              </div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                {article.title}
              </h2>
              <p className="text-gray-600 dark:text-gray-300 line-clamp-2">
                {article.excerpt}
              </p>
              <div className="flex flex-wrap gap-2 mt-3">
                {article.tags.slice(0, 3).map(tag => (
                  <span key={tag} className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                    {tag}
                  </span>
                ))}
              </div>
            </Link>
          ))}
        </div>

        {/* CTA */}
        <div className="mt-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-3">Ready to Start Learning?</h2>
          <p className="text-blue-100 mb-6">Go beyond articles. Get hands-on with 31 interactive lessons and 25 coding exercises.</p>
          <Link href="/lesson/1" className="inline-block px-8 py-3 bg-white text-blue-600 rounded-lg font-medium hover:bg-gray-100 transition">
            Start Learning Free
          </Link>
        </div>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/" className="hover:text-gray-900 dark:hover:text-white">Lessons</Link>
            <Link href="/practice" className="hover:text-gray-900 dark:hover:text-white">Practice</Link>
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms</Link>
          </div>
          <p className="text-center text-gray-400 dark:text-gray-500 text-xs mt-4">A product by {siteConfig.company}</p>
        </div>
      </footer>
    </div>
  );
}
