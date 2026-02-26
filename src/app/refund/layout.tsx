import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Refund Policy',
  description: 'StudyAI refund policy. 7-day money-back guarantee on premium course purchases.',
  alternates: { canonical: '/refund' },
  robots: { index: true, follow: true },
};

export default function RefundLayout({ children }: { children: React.ReactNode }) {
  return children;
}
