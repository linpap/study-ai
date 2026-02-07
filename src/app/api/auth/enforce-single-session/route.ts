import { NextResponse } from 'next/server'
import { createClient as createServerClient } from '@/lib/supabase/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

export async function POST() {
  try {
    const serverSupabase = await createServerClient()
    const { data: { session }, error: sessionError } = await serverSupabase.auth.getSession()

    if (sessionError || !session) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
    }

    if (!supabaseServiceKey) {
      return NextResponse.json({ error: 'Server configuration error' }, { status: 500 })
    }

    const adminClient = createClient(supabaseUrl, supabaseServiceKey)

    // Revoke all other sessions for this user, keeping only the current one
    const { error: signOutError } = await adminClient.auth.admin.signOut(
      session.access_token,
      'others'
    )

    if (signOutError) {
      console.error('Error enforcing single session:', signOutError)
      return NextResponse.json({ error: 'Failed to enforce single session' }, { status: 500 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Enforce single session error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
