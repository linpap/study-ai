'use client';

import { useState } from 'react';
import { Question } from '@/data/lessons';
import VoiceRecorder from './VoiceRecorder';

interface QuizProps {
  questions: Question[];
  onComplete: (score: number, total: number) => void;
}

interface Answer {
  questionId: string;
  answer: string;
  isCorrect: boolean | null;
  feedback: string;
  checked: boolean;
}

export default function Quiz({ questions, onComplete }: QuizProps) {
  const [answers, setAnswers] = useState<Record<string, Answer>>({});
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [checking, setChecking] = useState<string | null>(null);

  const question = questions[currentQuestion];

  const handleMCQAnswer = (option: string) => {
    const isCorrect = option === question.correctAnswer;
    setAnswers(prev => ({
      ...prev,
      [question.id]: {
        questionId: question.id,
        answer: option,
        isCorrect,
        feedback: isCorrect
          ? question.explanation
          : `${question.explanation}`,
        checked: true,
      },
    }));
  };

  const handleDescriptiveAnswer = (text: string) => {
    setAnswers(prev => ({
      ...prev,
      [question.id]: {
        questionId: question.id,
        answer: text,
        isCorrect: null,
        feedback: '',
        checked: false,
      },
    }));
  };

  const appendVoiceText = (text: string) => {
    const current = answers[question.id]?.answer || '';
    handleDescriptiveAnswer(current + ' ' + text);
  };

  const checkDescriptiveAnswer = async () => {
    setChecking(question.id);

    try {
      const response = await fetch('/api/check-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.question,
          answer: answers[question.id]?.answer || '',
          keywords: question.keywords,
          correctExplanation: question.explanation,
        }),
      });

      const data = await response.json();

      setAnswers(prev => ({
        ...prev,
        [question.id]: {
          ...prev[question.id],
          isCorrect: data.score >= 60,
          feedback: data.feedback,
          checked: true,
        },
      }));
    } catch (error) {
      console.error('Error checking answer:', error);
      // Fallback to keyword matching
      const answer = answers[question.id]?.answer?.toLowerCase() || '';
      const matchedKeywords = question.keywords?.filter(kw =>
        answer.includes(kw.toLowerCase())
      ) || [];
      const score = (matchedKeywords.length / (question.keywords?.length || 1)) * 100;

      setAnswers(prev => ({
        ...prev,
        [question.id]: {
          ...prev[question.id],
          isCorrect: score >= 50,
          feedback: score >= 50
            ? `Good answer! You mentioned ${matchedKeywords.length} out of ${question.keywords?.length} key concepts.`
            : `Your answer could be improved. Key concepts to include: ${question.keywords?.join(', ')}. ${question.explanation}`,
          checked: true,
        },
      }));
    }

    setChecking(null);
  };

  const goToNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
    } else {
      calculateResults();
    }
  };

  const goToPrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
    }
  };

  const calculateResults = () => {
    const correctCount = Object.values(answers).filter(a => a.isCorrect).length;
    setShowResults(true);
    onComplete(correctCount, questions.length);
  };

  if (showResults) {
    const correctCount = Object.values(answers).filter(a => a.isCorrect).length;
    const percentage = Math.round((correctCount / questions.length) * 100);

    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-center">Quiz Results</h2>

        <div className="text-center mb-8">
          <div className={`text-6xl font-bold mb-2 ${
            percentage >= 70 ? 'text-green-500' : percentage >= 50 ? 'text-yellow-500' : 'text-red-500'
          }`}>
            {percentage}%
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            {correctCount} out of {questions.length} correct
          </p>
          <p className="mt-2 text-lg">
            {percentage >= 70
              ? 'Excellent work! You have a solid understanding.'
              : percentage >= 50
              ? 'Good effort! Review the topics you missed.'
              : 'Keep studying! Review this lesson and try again.'}
          </p>
        </div>

        <div className="space-y-6">
          <h3 className="text-xl font-semibold">Review Your Answers</h3>
          {questions.map((q, idx) => {
            const answer = answers[q.id];
            return (
              <div
                key={q.id}
                className={`p-4 rounded-lg border-2 ${
                  answer?.isCorrect
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    : 'border-red-500 bg-red-50 dark:bg-red-900/20'
                }`}
              >
                <p className="font-medium mb-2">
                  {idx + 1}. {q.question}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Your answer: {answer?.answer || 'Not answered'}
                </p>
                {!answer?.isCorrect && (
                  <p className="text-sm mt-2 text-gray-700 dark:text-gray-200">
                    {answer?.feedback}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  const currentAnswer = answers[question.id];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
      {/* Progress */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-300 mb-2">
          <span>Question {currentQuestion + 1} of {questions.length}</span>
          <span>{question.type === 'mcq' ? 'Multiple Choice' : 'Descriptive'}</span>
        </div>
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Question */}
      <h3 className="text-xl font-semibold mb-6">{question.question}</h3>

      {/* Answer Input */}
      {question.type === 'mcq' ? (
        <div className="space-y-3">
          {question.options?.map((option, idx) => (
            <button
              key={idx}
              onClick={() => handleMCQAnswer(option)}
              disabled={currentAnswer?.checked}
              className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                currentAnswer?.checked
                  ? option === question.correctAnswer
                    ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                    : option === currentAnswer?.answer
                    ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                    : 'border-gray-200 dark:border-gray-600'
                  : currentAnswer?.answer === option
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/10'
              }`}
            >
              <span className="font-medium mr-3">
                {String.fromCharCode(65 + idx)}.
              </span>
              {option}
            </button>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          <textarea
            value={currentAnswer?.answer || ''}
            onChange={(e) => handleDescriptiveAnswer(e.target.value)}
            disabled={currentAnswer?.checked}
            placeholder="Type your answer here or use voice input..."
            className="w-full p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg min-h-[150px] focus:border-blue-500 focus:outline-none dark:bg-gray-700 dark:text-white"
          />
          <div className="flex gap-4">
            <VoiceRecorder
              onTranscript={appendVoiceText}
              disabled={currentAnswer?.checked}
            />
            {!currentAnswer?.checked && (
              <button
                onClick={checkDescriptiveAnswer}
                disabled={!currentAnswer?.answer || checking === question.id}
                className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {checking === question.id ? (
                  <>
                    <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Checking...
                  </>
                ) : (
                  'Check Answer'
                )}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Feedback */}
      {currentAnswer?.checked && (
        <div
          className={`mt-6 p-4 rounded-lg ${
            currentAnswer.isCorrect
              ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200'
              : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
          }`}
        >
          <p className="font-medium mb-1">
            {currentAnswer.isCorrect ? 'Correct!' : 'Not quite right'}
          </p>
          <p>{currentAnswer.feedback}</p>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <button
          onClick={goToPrevious}
          disabled={currentQuestion === 0}
          className="px-6 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={goToNext}
          disabled={!currentAnswer?.checked}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {currentQuestion === questions.length - 1 ? 'Finish Quiz' : 'Next Question'}
        </button>
      </div>
    </div>
  );
}
