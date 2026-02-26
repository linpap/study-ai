import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import { JsonLd } from "@/components/JsonLd";
import { siteConfig } from "@/data/site";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "StudyAI - Learn AI & Machine Learning",
    template: "%s | StudyAI",
  },
  description: siteConfig.description,
  keywords: ["AI", "artificial intelligence", "machine learning", "deep learning", "neural networks", "learn AI", "AI course", "ML course", "machine learning tutorial", "learn machine learning", "AI tutorial"],
  metadataBase: new URL(siteConfig.url),
  alternates: {
    canonical: '/',
  },
  icons: {
    icon: '/icon.svg',
    apple: '/icon.svg',
  },
  openGraph: {
    title: "StudyAI - Learn AI & Machine Learning",
    description: "Master AI & Machine Learning from zero to hero with 31 comprehensive lessons. Interactive quizzes, practice exercises, and hands-on learning.",
    type: "website",
    locale: "en_US",
    siteName: "StudyAI",
    url: siteConfig.url,
  },
  twitter: {
    card: "summary_large_image",
    title: "StudyAI - Learn AI & Machine Learning",
    description: "Master AI & Machine Learning from zero to hero with 31 comprehensive lessons. Interactive quizzes and hands-on learning.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

const GA_ID = process.env.NEXT_PUBLIC_GA4_ID;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {GA_ID && (
          <>
            <script async src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`} />
            <script
              dangerouslySetInnerHTML={{
                __html: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}');`,
              }}
            />
          </>
        )}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var mode = localStorage.getItem('theme');
                  if (mode === 'dark' || (!mode && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                    document.documentElement.classList.add('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
        <JsonLd data={{
          '@context': 'https://schema.org',
          '@type': 'WebSite',
          name: siteConfig.name,
          url: siteConfig.url,
          description: siteConfig.description,
          potentialAction: {
            '@type': 'SearchAction',
            target: {
              '@type': 'EntryPoint',
              urlTemplate: `${siteConfig.url}/?q={search_term_string}`,
            },
            'query-input': 'required name=search_term_string',
          },
        }} />
        <JsonLd data={{
          '@context': 'https://schema.org',
          '@type': 'Organization',
          name: siteConfig.company,
          url: siteConfig.url,
          logo: `${siteConfig.url}/icon.svg`,
          description: 'Technology company building innovative education and AI solutions.',
        }} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
