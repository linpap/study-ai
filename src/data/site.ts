export const siteConfig = {
  name: 'StudyAI',
  url: 'https://learnai.greensolz.com',
  description: 'Master AI & Machine Learning from zero to production. 31 comprehensive lessons, 25 hands-on coding exercises, 150+ quiz questions with AI evaluation.',
  tagline: 'Learn AI & Machine Learning',
  company: 'Greensolz',
  author: {
    name: 'Soumyajit Sarkar',
    role: 'Partner & CTO, Greensolz',
  },
  pricing: {
    currency: 'INR',
    amount: 1200,
    originalAmount: 2400,
    label: '₹1,200',
    type: 'one-time' as const,
  },
  stats: {
    lessons: 31,
    exercises: 25,
    quizQuestions: 150,
    hours: 40,
  },
  social: {
    email: 'support@greensolz.com',
  },
} as const;
