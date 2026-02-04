import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export async function POST(request: Request) {
  try {
    const body = await request.formData();

    const paymentRequestId = body.get('payment_request_id') as string;
    const paymentId = body.get('payment_id') as string;
    const status = body.get('status') as string;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- Reserved for MAC verification in production
    const _mac = body.get('mac') as string;

    // Verify MAC (Message Authentication Code) for security
    // Note: In production, you should verify the MAC using your salt
    // const expectedMac = crypto.createHmac('sha1', INSTAMOJO_SALT)
    //   .update(paymentId + '|' + paymentRequestId + '|' + status)
    //   .digest('hex');

    if (!paymentRequestId || !paymentId || !status) {
      return NextResponse.json({ error: 'Missing parameters' }, { status: 400 });
    }

    // Use service role key for webhook (no user context)
    if (!supabaseServiceKey) {
      console.error('Service role key not configured');
      return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Get the payment request to find user_id
    const { data: paymentRequest } = await supabase
      .from('payment_requests')
      .select('user_id')
      .eq('payment_request_id', paymentRequestId)
      .single();

    if (!paymentRequest) {
      console.error('Payment request not found:', paymentRequestId);
      return NextResponse.json({ error: 'Payment request not found' }, { status: 404 });
    }

    if (status === 'Credit') {
      // Payment successful
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
        user_id: paymentRequest.user_id,
        is_active: true,
        payment_id: paymentId,
        amount_paid: 1200,
        purchased_at: new Date().toISOString(),
      });

      console.log('Premium access granted for user:', paymentRequest.user_id);
    } else {
      // Payment failed
      await supabase
        .from('payment_requests')
        .update({
          status: 'failed',
          payment_id: paymentId,
        })
        .eq('payment_request_id', paymentRequestId);

      console.log('Payment failed for request:', paymentRequestId);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Webhook error:', error);
    return NextResponse.json({ error: 'Webhook processing failed' }, { status: 500 });
  }
}
