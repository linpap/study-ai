import { NextResponse } from 'next/server';

export async function GET() {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV,
    version: process.env.npm_package_version || '1.0.0',
    checks: {
      supabase: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      anthropic: !!process.env.ANTHROPIC_API_KEY,
      resend: !!process.env.RESEND_API_KEY,
    },
  };

  return NextResponse.json(health);
}
