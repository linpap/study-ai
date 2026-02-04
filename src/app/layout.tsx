import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "StudyAI - Learn Artificial Intelligence",
  description: "Comprehensive AI lessons from basics to building applications. Interactive quizzes with voice input support. 31 lessons covering machine learning, deep learning, neural networks, and more.",
  keywords: ["AI", "artificial intelligence", "machine learning", "deep learning", "neural networks", "learn AI", "AI course", "AI tutorial", "machine learning course"],
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://learnai.greensolz.com'),
  openGraph: {
    title: "StudyAI - Learn Artificial Intelligence",
    description: "Master AI from zero to hero with 31 comprehensive lessons. Interactive quizzes, practice exercises, and hands-on learning.",
    type: "website",
    locale: "en_US",
    siteName: "StudyAI",
  },
  twitter: {
    card: "summary_large_image",
    title: "StudyAI - Learn Artificial Intelligence",
    description: "Master AI from zero to hero with 31 comprehensive lessons. Interactive quizzes and hands-on learning.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
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
