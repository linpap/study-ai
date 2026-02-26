import { Metadata } from 'next';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';

export const metadata: Metadata = {
  title: 'Premium - Full AI & Machine Learning Course',
  description: `Get lifetime access to all ${siteConfig.stats.lessons} AI lessons, ${siteConfig.stats.exercises} coding exercises, and ${siteConfig.stats.quizQuestions}+ quiz questions for just ${siteConfig.pricing.label}. One-time payment, no subscription.`,
  alternates: { canonical: '/premium' },
  openGraph: {
    title: 'StudyAI Premium - Full Course Access',
    description: `Unlock all ${siteConfig.stats.lessons} lessons for ${siteConfig.pricing.label}. One-time payment, lifetime access.`,
    url: `${siteConfig.url}/premium`,
  },
};

export default function PremiumLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'Course',
        name: 'AI & Machine Learning Complete Course',
        description: `Master AI & Machine Learning from zero to production. ${siteConfig.stats.lessons} lessons, ${siteConfig.stats.exercises} exercises, ${siteConfig.stats.quizQuestions}+ quiz questions.`,
        provider: {
          '@type': 'Organization',
          name: siteConfig.company,
          sameAs: siteConfig.url,
        },
        offers: {
          '@type': 'Offer',
          price: siteConfig.pricing.amount,
          priceCurrency: siteConfig.pricing.currency,
          availability: 'https://schema.org/InStock',
          url: `${siteConfig.url}/premium`,
        },
        hasCourseInstance: {
          '@type': 'CourseInstance',
          courseMode: 'online',
          courseWorkload: `PT${siteConfig.stats.hours}H`,
        },
        numberOfCredits: siteConfig.stats.lessons,
        educationalLevel: 'Beginner to Advanced',
        teaches: 'Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, NLP, Computer Vision, Transformers, LLMs, RAG, Prompt Engineering, MLOps',
      }} />
      {children}
    </>
  );
}
