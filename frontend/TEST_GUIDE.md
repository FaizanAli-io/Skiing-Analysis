# Frontend Debugging Guide

## Current Issue: Blue Screen Only

You're seeing the blue gradient background but no content rendering.

## Steps to Fix:

### 1. Start the Backend
```bash
cd d:\upwork\Skiing-Analysis
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend (in a new terminal)
```bash
cd frontend
npm run dev
```

### 3. Open Browser Dev Tools
- Press F12 to open Developer Tools
- Go to the **Console** tab
- Look for any red errors

### 4. Common Errors and Fixes:

#### Error: "Failed to fetch" or "Network Error"
**Cause:** Backend not running or wrong URL
**Fix:** 
- Make sure backend is running on port 8000
- Check frontend/.env has: `VITE_API_BASE_URL=http://localhost:8000`
- Restart frontend after changing .env

#### Error: "Cannot find module" or "React is not defined"
**Cause:** Missing vite.config.js
**Fix:** Already created - restart dev server

#### Error: CORS policy blocking
**Cause:** Backend CORS not allowing frontend
**Fix:** Already configured in main.py - should work

#### Blank page, no errors
**Cause:** React not mounting
**Fix:** Check these:
1. Is `<div id="root"></div>` in index.html? ✅ (it is)
2. Is main.jsx being loaded? Check Network tab
3. Try adding this to the top of App.jsx:
   ```javascript
   console.log("App component loaded");
   ```

### 5. Quick Test
Add this to the very top of `src/App.jsx` after the imports:

```javascript
console.log("App.jsx loaded successfully");
console.log("API_BASE:", import.meta.env.VITE_API_BASE_URL);
```

Then reload the page and check the console.

### 6. If Still Blue Screen
The frontend is working but stuck on a route. Check:
- What does `window.location.pathname` show in console?
- Try manually going to: `http://localhost:5173/login`

### 7. Test Backend API
Open in browser: `http://localhost:8000/docs`
- Should show FastAPI Swagger UI
- If this doesn't work, backend isn't running

### Expected Flow:
1. Frontend loads → shows blue gradient ✅
2. React mounts → App component runs
3. useRoute() checks path → defaults to "/login"
4. AuthForm renders → login form appears
5. User sees: Logo, "Welcome back", email/password fields

## Files I Just Created:
- ✅ `frontend/vite.config.js` - Vite configuration
- ✅ `frontend/.env` - Environment variables
- ✅ This guide

## Next Steps:
1. Restart the frontend dev server
2. Check browser console
3. Report any errors you see
