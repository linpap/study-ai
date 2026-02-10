'use client'

import { createContext, useContext, useEffect, useState, ReactNode, useMemo, useCallback } from 'react'
import { User, AuthError, AuthChangeEvent, Session, AuthResponse } from '@supabase/supabase-js'
import { createClient } from '@/lib/supabase/client'

interface AuthContextType {
  user: User | null
  loading: boolean
  isPremium: boolean
  checkPremiumStatus: () => Promise<boolean>
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>
  signUp: (email: string, password: string) => Promise<{ error: AuthError | null; session: Session | null }>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [isPremium, setIsPremium] = useState(false)
  const supabase = useMemo(() => createClient(), [])

  const checkPremiumStatus = useCallback(async (): Promise<boolean> => {
    if (!supabase || !user) {
      setIsPremium(false)
      return false
    }

    try {
      const { data } = await supabase
        .from('user_premium')
        .select('is_active')
        .eq('user_id', user.id)
        .eq('is_active', true)
        .single()

      const premium = !!data?.is_active
      setIsPremium(premium)
      return premium
    } catch {
      setIsPremium(false)
      return false
    }
  }, [supabase, user])

  const migrateLocalProgress = useCallback(async (userId: string) => {
    if (!supabase) return

    try {
      const localProgress = localStorage.getItem('studyai-progress')
      if (!localProgress) return

      const progress = JSON.parse(localProgress)
      const entries = Object.entries(progress)

      if (entries.length === 0) return

      // Check if user already has progress in Supabase
      const { data: existingProgress } = await supabase
        .from('user_progress')
        .select('lesson_id')
        .eq('user_id', userId)

      const existingLessonIds = new Set(
        existingProgress?.map((p: { lesson_id: number }) => p.lesson_id) || []
      )

      // Only migrate lessons that don't exist in Supabase
      const progressToInsert = entries
        .filter(([lessonId]) => !existingLessonIds.has(parseInt(lessonId)))
        .map(([lessonId, data]) => {
          const progressData = data as {
            viewed?: boolean
            completed?: boolean
            score?: number
            completedAt?: string
          }
          return {
            user_id: userId,
            lesson_id: parseInt(lessonId),
            viewed: progressData.viewed || false,
            completed: progressData.completed || false,
            score: progressData.score || null,
            completed_at: progressData.completedAt || null,
          }
        })

      if (progressToInsert.length > 0) {
        await supabase.from('user_progress').insert(progressToInsert)
      }

      // Clear localStorage after successful migration
      localStorage.removeItem('studyai-progress')
    } catch (error) {
      console.error('Failed to migrate local progress:', error)
    }
  }, [supabase])

  useEffect(() => {
    // If Supabase is not configured, just set loading to false
    if (!supabase) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional early return when Supabase not configured
      setLoading(false)
      return
    }

    // Get initial session
    const getInitialSession = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      setUser(session?.user ?? null)
      setLoading(false)
    }

    getInitialSession()

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event: AuthChangeEvent, session: Session | null) => {
        setUser(session?.user ?? null)
        setLoading(false)

        // Migrate localStorage progress on sign in
        if (event === 'SIGNED_IN' && session?.user) {
          await migrateLocalProgress(session.user.id)
        }

        // Reset premium status on sign out
        if (event === 'SIGNED_OUT') {
          setIsPremium(false)
        }
      }
    )

    return () => {
      subscription.unsubscribe()
    }
  }, [supabase, migrateLocalProgress])

  // Check premium status when user changes
  useEffect(() => {
    if (user) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional check on user change
      checkPremiumStatus()
    }
  }, [user, checkPremiumStatus])

  const signIn = async (email: string, password: string) => {
    if (!supabase) {
      return { error: { message: 'Supabase is not configured' } as AuthError }
    }
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (!error) {
      // Revoke all other sessions (fire-and-forget)
      fetch('/api/auth/enforce-single-session', { method: 'POST' }).catch(console.error)
    }
    return { error }
  }

  const signUp = async (email: string, password: string) => {
    if (!supabase) {
      return { error: { message: 'Supabase is not configured' } as AuthError, session: null }
    }
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    return { error, session: data?.session ?? null }
  }

  const signOut = async () => {
    if (!supabase) return
    await supabase.auth.signOut()
  }

  return (
    <AuthContext.Provider value={{ user, loading, isPremium, checkPremiumStatus, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
