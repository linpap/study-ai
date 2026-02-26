export interface LearningPath {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
  lessonIds: number[];
  exerciseIds: number[];
  estimatedHours: number;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
}

export const learningPaths: LearningPath[] = [
  {
    id: 'ai-foundations',
    title: 'AI Foundations',
    description: 'Start your AI journey from scratch. Learn what AI is, how machine learning works, the essential math, and build your first models.',
    icon: '🧠',
    color: 'from-blue-500 to-cyan-500',
    lessonIds: [1, 2, 3, 4, 5, 10],
    exerciseIds: [1, 2, 3, 4, 5, 6, 7],
    estimatedHours: 6,
    difficulty: 'Beginner',
  },
  {
    id: 'classical-ml',
    title: 'Classical ML Practitioner',
    description: 'Master traditional machine learning algorithms. From decision trees to SVMs, learn the workhorses of industry ML.',
    icon: '📊',
    color: 'from-green-500 to-emerald-500',
    lessonIds: [6, 7, 8, 9],
    exerciseIds: [8, 9, 10, 11, 12, 13, 14, 15, 16],
    estimatedHours: 8,
    difficulty: 'Intermediate',
  },
  {
    id: 'deep-learning',
    title: 'Deep Learning Specialist',
    description: 'Dive into neural networks, CNNs, and advanced architectures. Build deep learning models for vision and beyond.',
    icon: '🔮',
    color: 'from-purple-500 to-pink-500',
    lessonIds: [10, 28, 29, 30, 31],
    exerciseIds: [17, 18, 19, 20, 25],
    estimatedHours: 8,
    difficulty: 'Intermediate',
  },
  {
    id: 'llm-engineer',
    title: 'LLM Engineer',
    description: 'Master large language models, prompt engineering, RAG systems, and NLP. Build production-ready LLM applications.',
    icon: '🤖',
    color: 'from-orange-500 to-red-500',
    lessonIds: [13, 18, 20, 21, 22, 23],
    exerciseIds: [26, 27, 28, 30, 32],
    estimatedHours: 10,
    difficulty: 'Advanced',
  },
  {
    id: 'production-ai',
    title: 'Production AI Engineer',
    description: 'Ship AI to production. Learn MLOps, model deployment, monitoring, scaling, and real-world engineering practices.',
    icon: '🚀',
    color: 'from-indigo-500 to-violet-500',
    lessonIds: [14, 19, 24, 25, 26, 27],
    exerciseIds: [26, 31],
    estimatedHours: 10,
    difficulty: 'Advanced',
  },
];

export function getPathById(id: string): LearningPath | undefined {
  return learningPaths.find(path => path.id === id);
}

export function getAllPathIds(): string[] {
  return learningPaths.map(path => path.id);
}
