import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { articles, getArticleBySlug, getAllArticleSlugs } from '@/data/articles';
import { siteConfig } from '@/data/site';
import { JsonLd } from '@/components/JsonLd';
import BlogPostClient from './BlogPostClient';

type Props = {
  params: Promise<{ slug: string }>;
};

export async function generateStaticParams() {
  return getAllArticleSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const article = getArticleBySlug(slug);

  if (!article) {
    return { title: 'Article Not Found' };
  }

  return {
    title: article.title,
    description: article.excerpt,
    alternates: {
      canonical: `/blog/${slug}`,
    },
    openGraph: {
      title: article.title,
      description: article.excerpt,
      type: 'article',
      url: `${siteConfig.url}/blog/${slug}`,
      publishedTime: article.date,
      authors: [article.author],
      tags: article.tags,
    },
    twitter: {
      card: 'summary_large_image',
      title: article.title,
      description: article.excerpt,
    },
  };
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params;
  const article = getArticleBySlug(slug);

  if (!article) {
    notFound();
  }

  const wordCount = article.content.replace(/<[^>]*>/g, '').split(/\s+/).length;

  return (
    <>
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: article.title,
        description: article.excerpt,
        author: {
          '@type': 'Person',
          name: article.author,
          jobTitle: 'Partner & CTO',
          worksFor: { '@type': 'Organization', name: siteConfig.company },
        },
        publisher: {
          '@type': 'Organization',
          name: siteConfig.company,
          logo: { '@type': 'ImageObject', url: `${siteConfig.url}/icon.svg` },
        },
        datePublished: article.date,
        dateModified: article.date,
        mainEntityOfPage: `${siteConfig.url}/blog/${slug}`,
        wordCount,
        articleSection: article.category,
        keywords: article.tags.join(', '),
      }} />
      <JsonLd data={{
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home', item: siteConfig.url },
          { '@type': 'ListItem', position: 2, name: 'Blog', item: `${siteConfig.url}/blog` },
          { '@type': 'ListItem', position: 3, name: article.title, item: `${siteConfig.url}/blog/${slug}` },
        ],
      }} />
      <BlogPostClient article={article} />
    </>
  );
}
