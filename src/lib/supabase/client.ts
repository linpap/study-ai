import { createBrowserClient } from '@supabase/ssr'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

export const isSupabaseConfigured = Boolean(
  supabaseUrl &&
  supabaseAnonKey &&
  supabaseUrl !== 'your_project_url' &&
  supabaseAnonKey !== 'your_anon_key'
)

let supabaseClient: ReturnType<typeof createBrowserClient> | null = null

export function createClient() {
  if (!isSupabaseConfigured) {
    return null
  }

  if (!supabaseClient) {
    supabaseClient = createBrowserClient(supabaseUrl!, supabaseAnonKey!)
  }

  return supabaseClient
}
