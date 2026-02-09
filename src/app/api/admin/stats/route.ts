import { NextResponse } from 'next/server';
import { createClient as createServerClient } from '@/lib/supabase/server';
import { createClient } from '@supabase/supabase-js';

const ADMIN_EMAIL = 'linpap@gmail.com';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export async function GET() {
  try {
    // Verify the caller is the admin via session
    const serverSupabase = await createServerClient();
    const { data: { user } } = await serverSupabase.auth.getUser();

    if (!user || user.email !== ADMIN_EMAIL) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
    }

    if (!supabaseServiceKey) {
      return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Count total users via auth admin API
    const { data: { users }, error: usersError } = await supabase.auth.admin.listUsers();
    if (usersError) {
      console.error('Error listing users:', usersError);
      return NextResponse.json({ error: 'Failed to list users' }, { status: 500 });
    }
    const totalUsers = users.length;

    // Count premium users
    const { count: premiumUsers, error: premiumError } = await supabase
      .from('user_premium')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true);

    if (premiumError) {
      console.error('Error counting premium users:', premiumError);
    }

    // Aggregate user_progress stats (bypasses RLS with service role)
    const { data: progressData, error: progressError } = await supabase
      .from('user_progress')
      .select('completed');

    if (progressError) {
      console.error('Error fetching progress:', progressError);
    }

    const totalAttempts = progressData?.length ?? 0;
    const completedLessons = progressData?.filter((p: { completed: boolean }) => p.completed).length ?? 0;

    return NextResponse.json({
      totalUsers,
      premiumUsers: premiumUsers ?? 0,
      completedLessons,
      totalAttempts,
    });
  } catch (error) {
    console.error('Admin stats error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
