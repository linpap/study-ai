import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Learning Paths - StudyAI',
  description: 'Structured learning paths to master AI & Machine Learning. Choose from AI Foundations, Classical ML, Deep Learning, LLM Engineering, and Production AI.',
};

export default function PathsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
