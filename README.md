# KCSE Results Capture Tool

FastAPI backend + Astro (Starlight) frontend to fetch and parse KCSE results from
the KNEC results portal. Supports single lookups, batch uploads from nominal roll
PDFs, and prefix-range search by index number.

## Features

- Single candidate lookup with parsed results.
- Batch processing from a nominal roll PDF.
- Prefix-range search to find a candidate using a base index + name.
- Full-page batch results view with filters, CSV export, and PDF print export.
- Frontend UI built with Astro/Starlight.

## Backend (FastAPI)

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install fastapi uvicorn httpx python-multipart
```

3. Ensure `pdftotext` is installed for PDF parsing:

```bash
sudo apt install poppler-utils
```

4. Run the API:

```bash
KNEC_VERIFY_SSL=false uvicorn main:app --reload
```

### Environment variables

- `KNEC_VERIFY_SSL` (default: `true`): set `false` to bypass SSL verification.
- `KNEC_CA_BUNDLE`: path to a custom CA bundle (overrides `KNEC_VERIFY_SSL`).
- `KNEC_ALLOWED_ORIGINS`: comma-separated list of allowed CORS origins.
  Default allows `http://localhost:4321` and `http://127.0.0.1:4321`.
- `KNEC_SEARCH_MAX` (default: `500`): max number of candidates for prefix search.

### API endpoints

- `POST /kcse/results`: raw HTML response from KNEC.
- `POST /kcse/results/parsed`: JSON parsed results for a single candidate.
- `POST /kcse/results/batch`: upload a nominal roll PDF and process candidates.
- `POST /kcse/results/search`: scan a suffix range for a base index + name.

### Example payloads

Single candidate:

```json
{
  "indexNumber": "29540204037",
  "name": "nelda",
  "consent": true
}
```

Prefix search:

```json
{
  "baseIndex": "29540102",
  "name": "nelda",
  "consent": true,
  "start": 1,
  "end": 50,
  "concurrency": 3
}
```

Batch upload:

```bash
curl -X POST http://127.0.0.1:8000/kcse/results/batch \
  -F "pdf=@NominallKcse.pdf" \
  -F "consent=true" \
  -F "limit=5" \
  -F "concurrency=3"
```

## Frontend (Astro)

### Setup

```bash
cd kcse-capture
npm install
npm run dev
```

Open `http://localhost:4321/`.

### Routes

- `/`: main capture page (single + batch).
- `/batch-results/`: full-page batch results view.
- `/search/`: prefix search page.

### Notes

- The frontend calls the backend directly, so CORS must allow the origin.
- Batch results are stored in `localStorage` for the full-page view.
- PDF export uses the browser print dialog ("Save as PDF").

## Privacy & Legal

- This tool is not affiliated with or endorsed by KNEC.
- The backend does not store or retain KNEC responses. Data is fetched on demand,
  returned to the requester, and discarded.
- Batch results are saved only in the browser (localStorage) for the full-page
  view. Use "Clear Saved" to remove them.
- You are responsible for authorization, consent, and compliance with local
  data protection requirements.
