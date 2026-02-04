'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function AboutPage() {
  const [darkMode, setDarkMode] = useState(false);
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
    if (savedDarkMode) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const faqs = [
    {
      question: "How is StudyAI different from other AI courses?",
      answer: "StudyAI combines interactive lessons, AI-evaluated quizzes, hands-on coding exercises, and voice input support. You don't just read - you practice and get instant feedback. Our 31 lessons cover everything from basics to building production AI applications."
    },
    {
      question: "Do I need programming experience?",
      answer: "Basic familiarity with programming concepts helps, but we start from the fundamentals. Our first few lessons cover the basics before diving into code. The practice exercises include hints and solutions to guide you."
    },
    {
      question: "How long does it take to complete all lessons?",
      answer: "The full curriculum is approximately 40+ hours of content. Most learners complete it in 4-8 weeks studying a few hours per week. You can learn at your own pace - your progress is saved automatically."
    },
    {
      question: "Are the first 3 lessons really free?",
      answer: "Yes! Lessons 1-3 are completely free with no credit card required. You can explore the platform, try the quizzes, and see if StudyAI is right for you before creating an account."
    },
    {
      question: "What topics are covered?",
      answer: "We cover: AI fundamentals, Machine Learning, Neural Networks, Deep Learning, NLP, Computer Vision, Transformers, LLMs, RAG systems, Prompt Engineering, MLOps, and building production AI applications."
    },
    {
      question: "Can I get a certificate?",
      answer: "We're working on adding completion certificates. For now, the real value is the knowledge and practical skills you gain - which you can demonstrate in interviews and projects."
    }
  ];

  const testimonials = [
    {
      name: "Priya Sharma",
      role: "Software Developer",
      content: "Finally, an AI course that doesn't just throw math at you. The interactive quizzes and practice exercises made concepts click. Built my first RAG app after lesson 21!",
      rating: 5
    },
    {
      name: "Rahul Mehta",
      role: "Data Analyst",
      content: "The voice input feature is a game-changer for learning. I can explain concepts out loud and get AI feedback. Went from zero ML knowledge to fine-tuning models in 6 weeks.",
      rating: 5
    },
    {
      name: "Ananya Krishnan",
      role: "Product Manager",
      content: "I needed to understand AI to work better with my engineering team. StudyAI gave me the technical foundation without overwhelming me. Highly recommend for non-engineers too.",
      rating: 5
    },
    {
      name: "Vikram Patel",
      role: "CS Student",
      content: "Better than my university AI course honestly. The hands-on coding exercises are practical and the explanations are clear. Worth every minute.",
      rating: 5
    }
  ];

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${darkMode ? 'dark' : ''}`}>
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <span className="text-xl font-bold text-gray-900 dark:text-white">StudyAI</span>
          </Link>
          <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition">
            Back to Lessons
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">About StudyAI</h1>

        {/* Mission Section */}
        <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">Our Mission</h2>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            StudyAI is dedicated to making artificial intelligence education accessible to everyone.
            We believe that understanding AI is becoming essential in today&apos;s world, and our platform
            provides comprehensive, interactive lessons that take you from the basics to building
            your own AI applications.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">31</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Lessons</div>
            </div>
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">150+</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Quiz Questions</div>
            </div>
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">25</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Practice Exercises</div>
            </div>
            <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">40+</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Hours of Content</div>
            </div>
          </div>
        </section>

        {/* Creator Section */}
        <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">Created By</h2>
          <div className="flex items-start gap-6">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white font-bold text-2xl">S</span>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Soumyajit S.</h3>
              <p className="text-blue-600 dark:text-blue-400 mb-3">AI Engineer & Educator</p>
              <p className="text-gray-600 dark:text-gray-300 mb-3">
                With over 20 years of experience in software development and 4+ years specializing in AI/ML,
                I&apos;ve built production AI systems for startups and enterprises. I created StudyAI because
                I saw too many courses that were either too theoretical or too shallow.
              </p>
              <p className="text-gray-600 dark:text-gray-300">
                My goal is to help you actually understand AI — not just memorize terms, but build intuition
                and practical skills you can apply immediately. Every lesson, quiz, and exercise is designed
                based on what I wish I had when I was learning.
              </p>
            </div>
          </div>
        </section>

        {/* What We Offer */}
        <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">What We Offer</h2>
          <ul className="text-gray-600 dark:text-gray-300 space-y-3">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>Comprehensive Lessons:</strong> From AI fundamentals to advanced topics like neural networks, transformers, and prompt engineering.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>Interactive Quizzes:</strong> Test your knowledge with multiple choice and descriptive questions, evaluated by AI for personalized feedback.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>Hands-on Practice:</strong> 25 coding exercises with real-time feedback, test cases, and solutions to build practical skills.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>Voice Input Support:</strong> Answer questions by speaking for a more natural learning experience.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>Progress Tracking:</strong> Your progress syncs across devices so you can learn anywhere.</span>
            </li>
          </ul>
        </section>

        {/* Testimonials */}
        <section className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">What Learners Say</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
                <div className="flex items-center gap-1 mb-3">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <svg key={i} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                  ))}
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-4 italic">&quot;{testimonial.content}&quot;</p>
                <div>
                  <p className="font-semibold text-gray-900 dark:text-white">{testimonial.name}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* FAQ Section */}
        <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">Frequently Asked Questions</h2>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div key={index} className="border-b border-gray-200 dark:border-gray-700 pb-4 last:border-0 last:pb-0">
                <button
                  onClick={() => setOpenFaq(openFaq === index ? null : index)}
                  className="w-full flex items-center justify-between text-left"
                >
                  <span className="font-medium text-gray-900 dark:text-white">{faq.question}</span>
                  <svg
                    className={`w-5 h-5 text-gray-500 transition-transform ${openFaq === index ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {openFaq === index && (
                  <p className="mt-3 text-gray-600 dark:text-gray-300">{faq.answer}</p>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Ready to Master AI?</h2>
          <p className="text-blue-100 mb-6">
            Start with 3 free lessons. No credit card required.
          </p>
          <Link
            href="/"
            className="inline-block px-8 py-3 bg-white text-blue-600 rounded-lg font-medium hover:bg-gray-100 transition"
          >
            Start Learning Free
          </Link>
        </section>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms of Service</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
