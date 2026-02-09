import { NextResponse } from 'next/server';
import { createClient as createServerClient } from '@/lib/supabase/server';
import { createClient } from '@supabase/supabase-js';

const ADMIN_EMAIL = 'linpap@gmail.com';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export async function POST(request: Request) {
  try {
    // Verify the caller is the admin via session
    const serverSupabase = await createServerClient();
    const { data: { user } } = await serverSupabase.auth.getUser();

    if (!user || user.email !== ADMIN_EMAIL) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
    }

    const { email } = await request.json();

    if (!email || typeof email !== 'string') {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 });
    }

    // Use service role key to bypass RLS and access auth.users
    if (!supabaseServiceKey) {
      return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Look up user by email
    const { data: { users }, error: listError } = await supabase.auth.admin.listUsers();

    if (listError) {
      console.error('Error listing users:', listError);
      return NextResponse.json({ error: 'Failed to look up users' }, { status: 500 });
    }

    const targetUser = users.find(
      (u) => u.email?.toLowerCase() === email.toLowerCase()
    );

    if (!targetUser) {
      return NextResponse.json(
        { error: `User with email "${email}" not found. They must register first.` },
        { status: 404 }
      );
    }

    // Upsert premium access
    const { error: upsertError } = await supabase.from('user_premium').upsert({
      user_id: targetUser.id,
      is_active: true,
      payment_id: 'admin-grant',
      amount_paid: 0,
      purchased_at: new Date().toISOString(),
    }, { onConflict: 'user_id' });

    if (upsertError) {
      console.error('Error granting premium:', upsertError);
      return NextResponse.json({ error: 'Failed to grant premium access' }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      message: `Premium access granted to ${email}`,
      user_id: targetUser.id,
    });
  } catch (error) {
    console.error('Grant premium error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
