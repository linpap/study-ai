import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';
import { getInstamojoHeaders, INSTAMOJO_BASE_URL } from '@/lib/instamojo';

export async function POST(request: Request) {
  try {
    const headers = getInstamojoHeaders();

    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json(
        { error: 'You must be logged in to make a purchase' },
        { status: 401 }
      );
    }

    // Check if user is already premium
    const { data: existingPremium } = await supabase
      .from('user_premium')
      .select('*')
      .eq('user_id', user.id)
      .eq('is_active', true)
      .single();

    if (existingPremium) {
      return NextResponse.json(
        { error: 'You already have premium access' },
        { status: 400 }
      );
    }

    const { redirect_url } = await request.json();
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://learnai.greensolz.com';

    // Create Instamojo payment request (API v1.1)
    const response = await fetch(`${INSTAMOJO_BASE_URL}/payment-requests/`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        purpose: 'StudyAI Premium - Lifetime Access',
        amount: '1200',
        buyer_name: user.user_metadata?.name || 'User',
        email: user.email || '',
        redirect_url: `${baseUrl}/api/payment/verify?user_id=${user.id}&redirect=${redirect_url || '/'}`,
        webhook: `${baseUrl}/api/payment/webhook`,
        allow_repeated_payments: 'False',
        send_email: 'False',
      }),
    });

    const data = await response.json();

    if (!data.success || !data.payment_request) {
      console.error('Instamojo error:', data);
      return NextResponse.json(
        { error: data.message || 'Failed to create payment request' },
        { status: 500 }
      );
    }

    const paymentRequest = data.payment_request;

    // Store payment request in database for tracking
    await supabase.from('payment_requests').insert({
      user_id: user.id,
      payment_request_id: paymentRequest.id,
      amount: 1200,
      status: 'pending',
    });

    return NextResponse.json({
      success: true,
      payment_url: paymentRequest.longurl,
    });
  } catch (error) {
    console.error('Create payment error:', error);
    return NextResponse.json(
      { error: 'Failed to create payment' },
      { status: 500 }
    );
  }
}
