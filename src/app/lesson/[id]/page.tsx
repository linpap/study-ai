import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { getLessonById, lessons } from '@/data/lessons';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';
import LessonClient from './LessonClient';

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateStaticParams() {
  return lessons.map((lesson) => ({ id: String(lesson.id) }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const lessonId = parseInt(id);
  const lesson = getLessonById(lessonId);

  if (!lesson) {
    return { title: 'Lesson Not Found' };
  }

  const currentIndex = lessons.findIndex(l => l.id === lessonId);
  const title = `${lesson.title} - Lesson ${currentIndex + 1}`;

  return {
    title,
    description: lesson.description,
    alternates: {
      canonical: `/lesson/${lessonId}`,
    },
    openGraph: {
      title: `${title} | StudyAI`,
      description: lesson.description,
      type: 'article',
      url: `${siteConfig.url}/lesson/${lessonId}`,
    },
  };
}

export default async function LessonPage({ params }: Props) {
  const { id } = await params;
  const lessonId = parseInt(id);
  const lesson = getLessonById(lessonId);

  if (!lesson) {
    notFound();
  }

  const currentIndex = lessons.findIndex(l => l.id === lessonId);

  return (
    <>
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          {
            '@type': 'ListItem',
            position: 1,
            name: 'Home',
            item: siteConfig.url,
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: 'Lessons',
            item: `${siteConfig.url}/`,
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: lesson.title,
            item: `${siteConfig.url}/lesson/${lessonId}`,
          },
        ],
      }} />
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'LearningResource',
        name: lesson.title,
        description: lesson.description,
        provider: {
          '@type': 'Organization',
          name: siteConfig.company,
        },
        educationalLevel: lesson.difficulty,
        timeRequired: `PT${parseInt(lesson.duration)}M`,
        isPartOf: {
          '@type': 'Course',
          name: 'AI & Machine Learning Complete Course',
          provider: { '@type': 'Organization', name: siteConfig.company },
        },
        position: currentIndex + 1,
      }} />
      <LessonClient lessonId={lessonId} />
    </>
  );
}
