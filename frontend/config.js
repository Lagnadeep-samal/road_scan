/* =========================================================================
   RoadScan runtime config
   -------------------------------------------------------------------------
   Leave this as an empty string when the backend serves this frontend
   itself (same origin — the default `python main.py` setup).

   When you deploy the frontend and backend SEPARATELY (e.g. frontend on
   Vercel, backend on Render), set this to your backend's full URL:

     window.ROADSCAN_API_BASE = "https://your-backend.onrender.com";

   No trailing slash.
   ========================================================================= */

window.ROADSCAN_API_BASE = "https://road-scan.onrender.com";