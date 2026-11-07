# Cookie Authentication Update

## What changed

- Register now creates a session and signs the user in automatically.
- Login and register set access, refresh, and CSRF cookies with `Set-Cookie`.
- Access and refresh cookies are `HttpOnly`; no authentication token is returned in JSON.
- Refresh reads the refresh token from its HttpOnly cookie and rotates the session.
- Logout revokes the current session (using access token or refresh token fallback)
  and expires all three cookies with matching cookie paths.
- User-protected endpoints accept the access cookie; legacy Bearer and project
  API-key integrations remain supported.
- Unsafe cookie-authenticated requests require the CSRF value in `X-CSRF-Token`.
- `GET /auth/csrf` returns the current CSRF value after a page reload.

## Environment

Local development:

```dotenv
AUTH_COOKIE_SECURE=false
AUTH_COOKIE_SAMESITE=lax
FRONTEND_ORIGINS=http://localhost:3000
```

Production:

```dotenv
AUTH_COOKIE_SECURE=true
AUTH_COOKIE_SAMESITE=lax
FRONTEND_ORIGINS=https://your-frontend.example.com
```

Use `SameSite=None` only when frontend and API are genuinely cross-site, and
only with `AUTH_COOKIE_SECURE=true`. Browser third-party-cookie restrictions
may still block cross-site cookies. A custom domain with frontend and API on
subdomains of the same registrable domain is preferable.

## Frontend request rules

- Always set `credentials: "include"` on fetch requests.
- After login/register, retain `csrf_token` from the JSON response in memory.
- After reload, call `GET /auth/csrf` with `credentials: "include"`.
- Send `X-CSRF-Token` on POST, PUT, PATCH, and DELETE requests made while
  cookie authentication is active.
- Do not read, store, or send access/refresh tokens from JavaScript.
- After refresh, replace the in-memory CSRF value with the returned value.
- Logout is a POST; send the CSRF header. The backend expires the cookies.

## Validation

The changed Python files pass syntax compilation. Cookie serialization was
checked for three HttpOnly cookies and three matching `Max-Age=0` deletion
headers. Full pytest could not be executed in the review environment because
project runtime dependencies (including `openai`, `faiss-cpu`,
`sentence-transformers`, and `pwdlib`) are not installed there. Run the
complete test suite in the project's Mac virtual environment before merging.
