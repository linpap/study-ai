export const siteConfig = {
  name: 'StudyAI',
  url: 'https://learnai.greensolz.com',
  description: 'Master AI & Machine Learning from zero to production. 31 comprehensive lessons, 33 hands-on coding exercises, 5 learning paths, and completion certificates.',
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
    exercises: 33,
    paths: 5,
    quizQuestions: 150,
    hours: 40,
  },
  social: {
    email: 'support@greensolz.com',
  },
} as const;
