import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Practice Exercises - AI & Machine Learning Coding',
  description: 'Hands-on AI and machine learning coding exercises. Practice implementing dot products, neural networks, gradient descent, NLP, and more with live code execution.',
  alternates: { canonical: '/practice' },
};

export default function PracticeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
