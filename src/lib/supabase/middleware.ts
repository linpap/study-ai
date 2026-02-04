import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || ''
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''

export async function updateSession(request: NextRequest) {
  let supabaseResponse = NextResponse.next({
    request,
  })

  // If Supabase is not configured, skip auth check
  if (!supabaseUrl || !supabaseAnonKey) {
    return supabaseResponse
  }

  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return request.cookies.getAll()
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value }) =>
          request.cookies.set(name, value)
        )
        supabaseResponse = NextResponse.next({
          request,
        })
        cookiesToSet.forEach(({ name, value, options }) =>
          supabaseResponse.cookies.set(name, value, options)
        )
      },
    },
  })

  // Refresh session if expired - important for Server Components
  const {
    data: { user },
  } = await supabase.auth.getUser()

  const url = request.nextUrl

  // Protected routes: Practice area requires authentication
  if (url.pathname.startsWith('/practice')) {
    if (!user) {
      const loginUrl = new URL('/auth/login', request.url)
      loginUrl.searchParams.set('redirect', url.pathname)
      return NextResponse.redirect(loginUrl)
    }
  }

  // Protected routes: Lesson 4 and above require authentication + premium
  const lessonMatch = url.pathname.match(/^\/lesson\/(\d+)$/)

  if (lessonMatch) {
    const lessonId = parseInt(lessonMatch[1], 10)
    const FREE_LESSONS = [1, 2, 3]

    if (!FREE_LESSONS.includes(lessonId)) {
      if (!user) {
        // Redirect to login with return URL
        const loginUrl = new URL('/auth/login', request.url)
        loginUrl.searchParams.set('redirect', url.pathname)
        return NextResponse.redirect(loginUrl)
      }

      // Check premium status for lessons 4+
      const { data: premiumData } = await supabase
        .from('user_premium')
        .select('is_active')
        .eq('user_id', user.id)
        .eq('is_active', true)
        .single()

      if (!premiumData?.is_active) {
        // Redirect to premium page
        const premiumUrl = new URL('/premium', request.url)
        premiumUrl.searchParams.set('redirect', url.pathname)
        return NextResponse.redirect(premiumUrl)
      }
    }
  }

  return supabaseResponse
}
