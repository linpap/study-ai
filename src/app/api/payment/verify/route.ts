import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

const INSTAMOJO_API_KEY = process.env.INSTAMOJO_API_KEY;
const INSTAMOJO_AUTH_TOKEN = process.env.INSTAMOJO_AUTH_TOKEN;
const INSTAMOJO_API_URL = process.env.INSTAMOJO_SANDBOX === 'true'
  ? 'https://test.instamojo.com/api/1.1'
  : 'https://www.instamojo.com/api/1.1';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const paymentRequestId = searchParams.get('payment_request_id');
    const paymentId = searchParams.get('payment_id');
    const userId = searchParams.get('user_id');
    const redirectPath = searchParams.get('redirect') || '/';

    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://learnai.greensolz.com';

    if (!paymentRequestId || !paymentId || !userId) {
      return NextResponse.redirect(`${baseUrl}/premium?error=missing_params`);
    }

    // Verify payment with Instamojo
    if (!INSTAMOJO_API_KEY || !INSTAMOJO_AUTH_TOKEN) {
      return NextResponse.redirect(`${baseUrl}/premium?error=config`);
    }

    const response = await fetch(
      `${INSTAMOJO_API_URL}/payment-requests/${paymentRequestId}/${paymentId}/`,
      {
        headers: {
          'X-Api-Key': INSTAMOJO_API_KEY,
          'X-Auth-Token': INSTAMOJO_AUTH_TOKEN,
        },
      }
    );

    const data = await response.json();

    if (!data.success) {
      console.error('Instamojo verification failed:', data);
      return NextResponse.redirect(`${baseUrl}/premium?error=verification_failed`);
    }

    const payment = data.payment_request;
    const paymentDetails = payment.payment;

    if (paymentDetails.status !== 'Credit') {
      return NextResponse.redirect(`${baseUrl}/premium?error=payment_failed`);
    }

    // Update database
    const supabase = await createClient();

    // Update payment request status
    await supabase
      .from('payment_requests')
      .update({
        status: 'completed',
        payment_id: paymentId,
        completed_at: new Date().toISOString(),
      })
      .eq('payment_request_id', paymentRequestId);

    // Grant premium access
    await supabase.from('user_premium').upsert({
      user_id: userId,
      is_active: true,
      payment_id: paymentId,
      amount_paid: 1200,
      purchased_at: new Date().toISOString(),
    });

    // Redirect to success page
    return NextResponse.redirect(`${baseUrl}/premium/success?redirect=${redirectPath}`);
  } catch (error) {
    console.error('Payment verification error:', error);
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://learnai.greensolz.com';
    return NextResponse.redirect(`${baseUrl}/premium?error=server_error`);
  }
}
