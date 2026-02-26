import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { getExerciseById, practiceExercises } from '@/data/practice-exercises';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';
import PracticeClient from './PracticeClient';

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateStaticParams() {
  return practiceExercises.map((ex) => ({ id: String(ex.id) }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const exerciseId = parseInt(id);
  const exercise = getExerciseById(exerciseId);

  if (!exercise) {
    return { title: 'Exercise Not Found' };
  }

  return {
    title: `${exercise.title} - Practice Exercise`,
    description: exercise.description,
    alternates: {
      canonical: `/practice/${exerciseId}`,
    },
    openGraph: {
      title: `${exercise.title} | StudyAI Practice`,
      description: exercise.description,
      url: `${siteConfig.url}/practice/${exerciseId}`,
    },
  };
}

export default async function ExercisePage({ params }: Props) {
  const { id } = await params;
  const exerciseId = parseInt(id);
  const exercise = getExerciseById(exerciseId);

  if (!exercise) {
    notFound();
  }

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
            name: 'Practice',
            item: `${siteConfig.url}/practice`,
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: exercise.title,
            item: `${siteConfig.url}/practice/${exerciseId}`,
          },
        ],
      }} />
      <PracticeClient exerciseId={exerciseId} />
    </>
  );
}
