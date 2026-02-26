import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { getPathById, learningPaths } from '@/data/learning-paths';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';
import PathClient from './PathClient';

type Props = {
  params: Promise<{ id: string }>;
};

export async function generateStaticParams() {
  return learningPaths.map((path) => ({ id: path.id }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const path = getPathById(id);

  if (!path) {
    return { title: 'Path Not Found' };
  }

  return {
    title: `${path.title} Learning Path - StudyAI`,
    description: path.description,
    alternates: {
      canonical: `/paths/${id}`,
    },
    openGraph: {
      title: `${path.title} | StudyAI Learning Paths`,
      description: path.description,
      url: `${siteConfig.url}/paths/${id}`,
    },
  };
}

export default async function PathPage({ params }: Props) {
  const { id } = await params;
  const path = getPathById(id);

  if (!path) {
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
            name: 'Learning Paths',
            item: `${siteConfig.url}/paths`,
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: path.title,
            item: `${siteConfig.url}/paths/${id}`,
          },
        ],
      }} />
      <PathClient pathId={id} />
    </>
  );
}
