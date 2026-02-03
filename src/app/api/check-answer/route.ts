import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { question, answer, keywords, correctExplanation } = await req.json();

    if (!answer || answer.trim().length < 10) {
      return NextResponse.json({
        score: 0,
        feedback: 'Please provide a more detailed answer.',
      });
    }

    // Check if Anthropic API key is available
    const apiKey = process.env.ANTHROPIC_API_KEY;

    if (apiKey) {
      // Use Claude to evaluate the answer
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 500,
          messages: [
            {
              role: 'user',
              content: `You are an AI tutor evaluating a student's answer. Be encouraging but accurate.

Question: ${question}

Expected key concepts to cover: ${keywords?.join(', ')}

Model answer explanation: ${correctExplanation}

Student's answer: ${answer}

Evaluate the student's answer and respond in this exact JSON format:
{
  "score": <number 0-100>,
  "feedback": "<constructive feedback explaining what was good and what could be improved>"
}

Score guide:
- 80-100: Excellent, covers most key concepts well
- 60-79: Good, covers some key concepts but missing others
- 40-59: Partial understanding, needs more detail
- 0-39: Missing most key concepts

Respond ONLY with the JSON, no other text.`,
            },
          ],
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const content = data.content[0].text;

        try {
          const result = JSON.parse(content);
          return NextResponse.json({
            score: result.score,
            feedback: result.feedback,
          });
        } catch {
          // If JSON parsing fails, extract what we can
          const scoreMatch = content.match(/score["\s:]+(\d+)/i);
          const score = scoreMatch ? parseInt(scoreMatch[1]) : 50;
          return NextResponse.json({
            score,
            feedback: content.replace(/[{}"]/g, '').trim(),
          });
        }
      }
    }

    // Fallback: keyword-based evaluation
    const answerLower = answer.toLowerCase();
    const matchedKeywords = keywords?.filter((kw: string) =>
      answerLower.includes(kw.toLowerCase())
    ) || [];

    const keywordScore = keywords?.length
      ? (matchedKeywords.length / keywords.length) * 100
      : 50;

    // Length bonus (up to 20 points)
    const lengthBonus = Math.min(answer.split(' ').length / 50, 1) * 20;

    const totalScore = Math.min(keywordScore + lengthBonus, 100);

    let feedback = '';
    if (totalScore >= 70) {
      feedback = `Good answer! You covered ${matchedKeywords.length} out of ${keywords?.length || 0} key concepts: ${matchedKeywords.join(', ')}.`;
    } else if (totalScore >= 50) {
      const missingKeywords = keywords?.filter(
        (kw: string) => !matchedKeywords.includes(kw)
      );
      feedback = `Partial answer. You mentioned: ${matchedKeywords.join(', ') || 'few key concepts'}. Consider also discussing: ${missingKeywords?.slice(0, 3).join(', ')}.`;
    } else {
      feedback = `Your answer needs more detail. Key concepts to include: ${keywords?.slice(0, 4).join(', ')}. ${correctExplanation}`;
    }

    return NextResponse.json({
      score: Math.round(totalScore),
      feedback,
    });
  } catch (error) {
    console.error('Error checking answer:', error);
    return NextResponse.json(
      { error: 'Failed to check answer', score: 50, feedback: 'Unable to evaluate. Please try again.' },
      { status: 500 }
    );
  }
}
