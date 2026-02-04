import { Resend } from 'resend';
import { NextResponse } from 'next/server';

const resend = new Resend(process.env.RESEND_API_KEY);

export async function POST(request: Request) {
  try {
    const { email } = await request.json();

    if (!email) {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 });
    }

    const { data, error } = await resend.emails.send({
      from: 'StudyAI <hello@learnai.greensolz.com>',
      to: email,
      subject: 'Welcome to StudyAI! 🎓',
      html: `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
          </head>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="text-align: center; margin-bottom: 30px;">
              <div style="display: inline-block; width: 60px; height: 60px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 12px; line-height: 60px; margin-bottom: 10px;">
                <span style="color: white; font-size: 28px; font-weight: bold;">S</span>
              </div>
              <h1 style="margin: 10px 0; color: #1f2937;">Welcome to StudyAI!</h1>
            </div>

            <p style="font-size: 16px; color: #4b5563;">Hi there!</p>

            <p style="font-size: 16px; color: #4b5563;">
              Thank you for joining StudyAI. You're now part of a community learning artificial intelligence from the ground up.
            </p>

            <div style="background: linear-gradient(135deg, #eff6ff, #f0fdf4); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 8px; margin: 25px 0;">
              <h3 style="margin: 0 0 10px 0; color: #1f2937;">What's included:</h3>
              <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
                <li>10 comprehensive AI lessons</li>
                <li>Interactive quizzes with AI evaluation</li>
                <li>Hands-on coding exercises</li>
                <li>Voice input support</li>
                <li>Progress tracking across devices</li>
              </ul>
            </div>

            <p style="font-size: 16px; color: #4b5563;">
              Your first 3 lessons are completely free. Dive in and start your AI learning journey today!
            </p>

            <div style="text-align: center; margin: 30px 0;">
              <a href="https://study-ai-six.vercel.app" style="display: inline-block; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                Start Learning
              </a>
            </div>

            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">

            <p style="font-size: 14px; color: #9ca3af; text-align: center;">
              Questions? Reply to this email or visit our <a href="https://study-ai-six.vercel.app/contact" style="color: #3b82f6;">contact page</a>.
            </p>

            <p style="font-size: 14px; color: #9ca3af; text-align: center;">
              Happy learning!<br>
              The StudyAI Team
            </p>
          </body>
        </html>
      `,
    });

    if (error) {
      console.error('Resend error:', error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ success: true, data });
  } catch (error) {
    console.error('Send welcome email error:', error);
    return NextResponse.json({ error: 'Failed to send email' }, { status: 500 });
  }
}
