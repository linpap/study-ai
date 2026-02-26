import { Metadata } from 'next';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';

export const metadata: Metadata = {
  title: 'About StudyAI - AI & Machine Learning Education Platform',
  description: `StudyAI by ${siteConfig.company} offers ${siteConfig.stats.lessons} comprehensive AI lessons, ${siteConfig.stats.exercises} coding exercises, and ${siteConfig.stats.quizQuestions}+ quiz questions. Learn AI from basics to production.`,
  alternates: { canonical: '/about' },
  openGraph: {
    title: 'About StudyAI',
    description: 'Learn about our mission to make AI education accessible to everyone.',
    url: `${siteConfig.url}/about`,
  },
};

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

export default function AboutLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: faqs.map(faq => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: faq.answer,
          },
        })),
      }} />
      {children}
    </>
  );
}
