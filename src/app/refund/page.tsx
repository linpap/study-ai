'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function RefundPage() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional sync with localStorage on mount
    setDarkMode(savedDarkMode);
    if (savedDarkMode) {
      document.documentElement.classList.add('dark');
    }
  }, []);

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
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Refund & Cancellation Policy</h1>
        <p className="text-gray-500 dark:text-gray-400 mb-8">Last updated: February 2026</p>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm space-y-8">
          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">1. Digital Product Policy</h2>
            <p className="text-gray-600 dark:text-gray-300">
              StudyAI Premium is a digital product that provides instant access to educational content.
              Due to the nature of digital products, all sales are generally considered final once access
              has been granted.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">2. Refund Eligibility</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-3">You may be eligible for a refund if:</p>
            <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2 ml-4">
              <li>You request a refund within <strong>7 days</strong> of purchase</li>
              <li>You have not completed more than 25% of the premium content</li>
              <li>You experienced technical issues that prevented access to the content</li>
              <li>The product was significantly different from what was advertised</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">3. Non-Refundable Cases</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-3">Refunds will not be provided if:</p>
            <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-2 ml-4">
              <li>More than 7 days have passed since purchase</li>
              <li>You have completed more than 25% of the premium lessons</li>
              <li>You simply changed your mind after accessing the content</li>
              <li>You violated our Terms of Service</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">4. How to Request a Refund</h2>
            <p className="text-gray-600 dark:text-gray-300 mb-3">To request a refund:</p>
            <ol className="list-decimal list-inside text-gray-600 dark:text-gray-300 space-y-2 ml-4">
              <li>Email us at <a href="mailto:support@greensolz.com" className="text-blue-500 hover:underline">support@greensolz.com</a></li>
              <li>Include your registered email address</li>
              <li>Provide your payment receipt or transaction ID</li>
              <li>Explain the reason for your refund request</li>
            </ol>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">5. Refund Processing</h2>
            <p className="text-gray-600 dark:text-gray-300">
              Once your refund request is approved, we will process the refund within <strong>5-7 business days</strong>.
              The refund will be credited to your original payment method. Please note that your bank or
              payment provider may take additional time to reflect the refund in your account.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">6. Cancellation</h2>
            <p className="text-gray-600 dark:text-gray-300">
              StudyAI Premium is a one-time purchase, not a subscription. There is no recurring billing
              to cancel. Once purchased, you have lifetime access to the premium content available at the
              time of purchase.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">7. Contact Us</h2>
            <p className="text-gray-600 dark:text-gray-300">
              If you have any questions about our refund policy, please{' '}
              <Link href="/contact" className="text-blue-500 hover:underline">contact us</Link> or
              email <a href="mailto:support@greensolz.com" className="text-blue-500 hover:underline">support@greensolz.com</a>.
            </p>
          </section>
        </div>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <Link href="/about" className="hover:text-gray-900 dark:hover:text-white">About</Link>
            <Link href="/contact" className="hover:text-gray-900 dark:hover:text-white">Contact</Link>
            <Link href="/privacy" className="hover:text-gray-900 dark:hover:text-white">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-gray-900 dark:hover:text-white">Terms of Service</Link>
            <Link href="/refund" className="hover:text-gray-900 dark:hover:text-white">Refund Policy</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
