# StudyAI Launch Status
**Last Updated:** 2026-02-05

## Platform
- **URL:** https://learnai.greensolz.com
- **Domain:** learnai.greensolz.com
- **Repo:** https://github.com/linpap/study-ai

---

## Completed Items

| Item | Status | Date |
|------|--------|------|
| ESLint/TypeScript errors | ✅ Fixed | Feb 5 |
| Error pages (404, error.tsx) | ✅ Added | Feb 5 |
| Health check endpoint | ✅ `/api/health` | Feb 5 |
| SEO meta tags (OG, Twitter) | ✅ Added | Feb 5 |
| Dynamic lesson count | ✅ 31 lessons | Feb 5 |
| ML branding | ✅ "AI & Machine Learning" | Feb 5 |
| About page | ✅ 20 years experience | Feb 5 |
| Admin email env var | ✅ `NEXT_PUBLIC_ADMIN_EMAIL` | Feb 5 |
| Supabase RLS policies | ✅ All 3 tables secured | Feb 5 |
| `.env.example` | ✅ Created | Feb 5 |

---

## Waiting / In Progress

| Item | Status | Notes |
|------|--------|-------|
| Resend email DNS | ⏳ Propagating | Check in 2-4 hrs, verify in Resend dashboard |

---

## Pending (Before Full Production)

| Item | Priority | Notes |
|------|----------|-------|
| Payment webhook MAC verification | 🔴 CRITICAL | Need Instamojo Salt key to implement |
| Rate limiting on APIs | 🟡 Medium | Contact form, payment endpoints |

---

## Database Tables (Supabase)

| Table | RLS Enabled | Policies |
|-------|-------------|----------|
| user_progress | ✅ Yes | SELECT/INSERT/UPDATE own data |
| user_premium | ✅ Yes | SELECT own data only |
| payment_requests | ✅ Yes | SELECT/INSERT own data |

---

## Environment Variables (Vercel)

| Variable | Status |
|----------|--------|
| NEXT_PUBLIC_SUPABASE_URL | ✅ Set |
| NEXT_PUBLIC_SUPABASE_ANON_KEY | ✅ Set |
| SUPABASE_SERVICE_ROLE_KEY | ✅ Set |
| ANTHROPIC_API_KEY | ✅ Set |
| RESEND_API_KEY | ✅ Set |
| INSTAMOJO_API_KEY | ✅ Set |
| INSTAMOJO_AUTH_TOKEN | ✅ Set |
| NEXT_PUBLIC_ADMIN_EMAIL | ✅ Set (linpap@gmail.com) |
| ADMIN_EMAIL | ✅ Set (linpap@gmail.com) |

---

## Ready For

- ✅ Soft launch (free content)
- ✅ User registration/login
- ✅ Lessons 1-3 (free)
- ✅ Practice exercises
- ⚠️ Premium content (works but payment webhook not secured)
- ❌ Real payments (DO NOT until webhook MAC is implemented)

---

## Next Steps

1. Check Resend DNS verification (should be green after propagation)
2. Get Instamojo Salt key and implement webhook MAC verification
3. Test full payment flow in sandbox mode
4. Launch!
