import asyncio
import os
import re
import subprocess
import tempfile
from html.parser import HTMLParser

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

app = FastAPI()


def _allowed_origins() -> list[str]:
    raw = os.getenv("KNEC_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:4321", "http://127.0.0.1:4321"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResultsRequest(BaseModel):
    indexNumber: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    consent: bool = True


class SearchRequest(BaseModel):
    baseIndex: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    consent: bool = True
    start: int = Field(1, ge=0)
    end: int = Field(999, ge=0)
    concurrency: int = Field(3, ge=1, le=10)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_name(name: str) -> str:
    return _normalize_text(name).upper()


class KnecResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_summary = ""
        self.mean_grade = ""
        self.alert_messages: list[str] = []
        self.rows: list[list[str]] = []

        self._candidate_buffer: list[str] = []
        self._mean_buffer: list[str] = []
        self._alert_buffer: list[str] = []
        self._header_buffer: list[str] = []
        self._cell_buffer: list[str] = []

        self._collect_candidate = False
        self._collect_mean = False
        self._collect_alert = False
        self._collect_header = False
        self._collect_cell = False

        self._alert_depth = 0
        self._in_table = False
        self._in_thead = False
        self._in_tbody = False
        self._headers: list[str] = []
        self._is_results_table = False
        self._current_row: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = attrs_dict.get("class", "").split()

        if tag == "h6" and "text-muted" in classes:
            self._collect_candidate = True
            self._candidate_buffer = []

        if tag == "span" and "ms-2" in classes and "fw-semibold" in classes:
            self._collect_mean = True
            self._mean_buffer = []

        if tag == "div":
            if self._collect_alert:
                self._alert_depth += 1
            else:
                if "alert" in classes or any(cls.startswith("alert-") for cls in classes):
                    self._collect_alert = True
                    self._alert_depth = 1
                    self._alert_buffer = []

        if tag == "table":
            self._in_table = True
            self._headers = []
            self._is_results_table = False

        if self._in_table and tag == "thead":
            self._in_thead = True

        if self._in_table and tag == "tbody":
            self._in_tbody = True

        if self._in_table and self._in_thead and tag == "th":
            self._collect_header = True
            self._header_buffer = []

        if self._in_table and self._in_tbody and tag == "tr":
            self._current_row = []

        if self._in_table and self._in_tbody and tag == "td":
            self._collect_cell = True
            self._cell_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h6" and self._collect_candidate:
            self.candidate_summary = _normalize_text("".join(self._candidate_buffer))
            self._collect_candidate = False

        if tag == "span" and self._collect_mean:
            self.mean_grade = _normalize_text("".join(self._mean_buffer))
            self._collect_mean = False

        if tag == "div" and self._collect_alert:
            self._alert_depth -= 1
            if self._alert_depth <= 0:
                message = _normalize_text("".join(self._alert_buffer))
                if message:
                    self.alert_messages.append(message)
                self._collect_alert = False

        if tag == "th" and self._collect_header:
            header = _normalize_text("".join(self._header_buffer))
            if header:
                self._headers.append(header)
            self._collect_header = False

        if tag == "thead" and self._in_thead:
            self._in_thead = False
            self._is_results_table = any(
                header.upper() == "SUBJECT NAME" for header in self._headers
            )

        if tag == "td" and self._collect_cell:
            cell = _normalize_text("".join(self._cell_buffer))
            if self._is_results_table:
                self._current_row.append(cell)
            self._collect_cell = False

        if tag == "tr" and self._in_tbody:
            if self._is_results_table and self._current_row:
                self.rows.append(self._current_row)

        if tag == "tbody" and self._in_tbody:
            self._in_tbody = False

        if tag == "table" and self._in_table:
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._collect_candidate:
            self._candidate_buffer.append(data)
        if self._collect_mean:
            self._mean_buffer.append(data)
        if self._collect_alert:
            self._alert_buffer.append(data)
        if self._collect_header:
            self._header_buffer.append(data)
        if self._collect_cell:
            self._cell_buffer.append(data)


def _parse_candidate_summary(summary: str) -> dict:
    index_number = None
    name = None
    school = None

    if summary:
        if " - " in summary:
            index_number, remainder = summary.split(" - ", 1)
        else:
            remainder = summary
        if "||" in remainder:
            name, school = [part.strip() for part in remainder.split("||", 1)]
        else:
            name = remainder.strip()

    return {
        "index_number": index_number.strip() if index_number else None,
        "name": name if name else None,
        "school": school if school else None,
    }


def _parse_knec_results(html: str) -> dict:
    parser = KnecResultsParser()
    parser.feed(html)
    candidate = _parse_candidate_summary(parser.candidate_summary)
    subjects: list[dict] = []

    for row in parser.rows:
        if len(row) < 4:
            continue
        subjects.append(
            {
                "code": row[1],
                "name": row[2],
                "grade": row[3],
            }
        )

    parsed = {
        "candidate": candidate,
        "mean_grade": parser.mean_grade or None,
        "subjects": subjects,
    }
    if parser.alert_messages:
        parsed["alerts"] = parser.alert_messages
    return parsed


def _knec_client_kwargs() -> dict:
    verify_env = os.getenv("KNEC_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_env not in ("0", "false", "no", "off")
    ca_bundle = os.getenv("KNEC_CA_BUNDLE", "").strip() or None
    verify_setting = ca_bundle if ca_bundle else verify_ssl
    headers = {"User-Agent": "Mozilla/5.0"}
    return {
        "follow_redirects": True,
        "timeout": 15.0,
        "verify": verify_setting,
        "headers": headers,
    }


async def _fetch_knec_results(
    payload: ResultsRequest,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    url = "https://results.knec.ac.ke/Home/CheckResults"
    data = {
        "indexNumber": payload.indexNumber,
        "name": payload.name,
        "consent": "true" if payload.consent else "false",
    }
    try:
        if client is None:
            async with httpx.AsyncClient(**_knec_client_kwargs()) as session:
                response = await session.post(url, data=data)
        else:
            response = await client.post(url, data=data)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream returned {response.status_code}",
        )

    return response


_NOMINAL_LINE_RE = re.compile(
    r"^\s*(\d{11})\s+([MF])\s+(\d+)\s+([A-Z0-9'\- ]+?)\s+(?=\d{3}\s)"
)


def _parse_nominal_roll_text(text: str) -> list[dict]:
    students: list[dict] = []
    seen: set[str] = set()

    for line in text.splitlines():
        match = _NOMINAL_LINE_RE.match(line)
        if not match:
            continue
        index_number, gender, birth_no, name = match.groups()
        index_number = index_number.strip()
        if index_number in seen:
            continue
        seen.add(index_number)
        students.append(
            {
                "index_number": index_number,
                "name": _normalize_text(name),
                "gender": gender,
                "birth_no": birth_no.strip(),
            }
        )

    return students


def _pdftotext_layout(pdf_path: str) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="pdftotext not available; install poppler-utils",
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or "pdftotext failed").strip()
        raise HTTPException(status_code=500, detail=message) from exc

    return result.stdout


def _extract_nominal_roll_from_pdf(pdf_bytes: bytes) -> list[dict]:
    if not pdf_bytes:
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        text = _pdftotext_layout(temp_file.name)

    return _parse_nominal_roll_text(text)


def _name_matches(candidate_name: str | None, query_name: str) -> bool:
    if not candidate_name:
        return False
    candidate_norm = _normalize_name(candidate_name)
    query_norm = _normalize_name(query_name)
    return query_norm in candidate_norm


def _build_index_candidates(base_index: str, start: int, end: int) -> list[str]:
    base = base_index.strip()
    if not base:
        raise HTTPException(status_code=400, detail="Base index number is required")

    missing = 11 - len(base)
    if missing < 0:
        raise HTTPException(
            status_code=400,
            detail="Base index is longer than 11 digits",
        )

    if start > end:
        raise HTTPException(status_code=400, detail="Start must be <= end")

    if missing == 0:
        return [base]

    max_candidates = int(os.getenv("KNEC_SEARCH_MAX", "500"))
    total = end - start + 1
    if total > max_candidates:
        raise HTTPException(
            status_code=400,
            detail=f"Range too large; max {max_candidates} candidates allowed",
        )

    max_value = (10**missing) - 1
    if end > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"End exceeds {max_value} for {missing}-digit suffix",
        )

    width = missing
    return [f"{base}{suffix:0{width}d}" for suffix in range(start, end + 1)]


async def _search_candidate_result(
    index_number: str,
    query_name: str,
    consent: bool,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = ResultsRequest(
        indexNumber=index_number,
        name=query_name,
        consent=consent,
    )
    try:
        async with semaphore:
            response = await _fetch_knec_results(payload, client=client)
        parsed = _parse_knec_results(response.text)
        candidate = parsed["candidate"]
        if not candidate.get("index_number"):
            candidate["index_number"] = index_number
        if not candidate.get("name"):
            candidate["name"] = query_name

        if not parsed["subjects"]:
            return {"status": "not_found", "index_number": index_number}

        if not _name_matches(candidate.get("name"), query_name):
            return {
                "status": "name_mismatch",
                "index_number": index_number,
                "candidate": candidate,
            }

        return {
            "status": "ok",
            "index_number": index_number,
            "result": parsed,
        }
    except HTTPException as exc:
        return {
            "status": "error",
            "index_number": index_number,
            "error": exc.detail,
        }
    except Exception as exc:  # pragma: no cover - defensive for batch runs
        return {
            "status": "error",
            "index_number": index_number,
            "error": f"Unexpected error: {exc}",
        }


async def _fetch_student_result(
    student: dict,
    consent: bool,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = ResultsRequest(
        indexNumber=student["index_number"],
        name=student["name"],
        consent=consent,
    )
    try:
        async with semaphore:
            response = await _fetch_knec_results(payload, client=client)
        parsed = _parse_knec_results(response.text)
        candidate = parsed["candidate"]
        if not candidate.get("index_number"):
            candidate["index_number"] = student["index_number"]
        if not candidate.get("name"):
            candidate["name"] = student["name"]

        if not parsed["subjects"]:
            return {
                "index_number": student["index_number"],
                "name": student["name"],
                "status": "not_found",
                "alerts": parsed.get("alerts") or [],
            }

        return {
            "index_number": student["index_number"],
            "name": student["name"],
            "status": "ok",
            "result": parsed,
        }
    except HTTPException as exc:
        return {
            "index_number": student["index_number"],
            "name": student["name"],
            "status": "error",
            "error": exc.detail,
        }
    except Exception as exc:  # pragma: no cover - defensive for batch runs
        return {
            "index_number": student["index_number"],
            "name": student["name"],
            "status": "error",
            "error": f"Unexpected error: {exc}",
        }


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/kcse/results")
async def kcse_results(payload: ResultsRequest) -> Response:
    response = await _fetch_knec_results(payload)
    media_type = response.headers.get("content-type", "text/html")
    return Response(content=response.text, media_type=media_type)


@app.post("/kcse/results/parsed")
async def kcse_results_parsed(payload: ResultsRequest) -> dict:
    response = await _fetch_knec_results(payload)
    parsed = _parse_knec_results(response.text)
    candidate = parsed["candidate"]
    if not candidate.get("index_number"):
        candidate["index_number"] = payload.indexNumber
    if not candidate.get("name"):
        candidate["name"] = payload.name

    if not parsed["subjects"]:
        alert_message = None
        alerts = parsed.get("alerts") or []
        if alerts:
            alert_message = alerts[0]
        raise HTTPException(
            status_code=404,
            detail=alert_message or "Results not found",
        )

    return parsed


@app.post("/kcse/results/batch")
async def kcse_results_batch(
    pdf: UploadFile = File(...),
    consent: bool = Form(True),
    limit: int | None = Form(None),
    concurrency: int = Form(3),
) -> dict:
    if pdf.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Expected a PDF upload")

    pdf_bytes = await pdf.read()
    students = _extract_nominal_roll_from_pdf(pdf_bytes)
    if not students:
        raise HTTPException(status_code=400, detail="No students found in PDF")

    if limit is not None:
        if limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be positive")
        students = students[:limit]

    concurrency = max(1, min(concurrency, 10))
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(**_knec_client_kwargs()) as client:
        tasks = [
            _fetch_student_result(student, consent, client, semaphore)
            for student in students
        ]
        results = await asyncio.gather(*tasks)

    return {
        "total": len(students),
        "processed": len(results),
        "results": results,
    }


@app.post("/kcse/results/search")
async def kcse_results_search(payload: SearchRequest) -> dict:
    candidates = _build_index_candidates(
        payload.baseIndex,
        payload.start,
        payload.end,
    )
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates to search")

    concurrency = max(1, min(payload.concurrency, 10))
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(**_knec_client_kwargs()) as client:
        tasks = [
            _search_candidate_result(
                index_number,
                payload.name,
                payload.consent,
                client,
                semaphore,
            )
            for index_number in candidates
        ]
        results = await asyncio.gather(*tasks)

    found = [item for item in results if item["status"] == "ok"]
    name_mismatch = [item for item in results if item["status"] == "name_mismatch"]
    not_found = [item for item in results if item["status"] == "not_found"]
    errors = [item for item in results if item["status"] == "error"]

    return {
        "searched": len(candidates),
        "found": len(found),
        "results": found,
        "summary": {
            "not_found": len(not_found),
            "name_mismatch": len(name_mismatch),
            "errors": len(errors),
        },
        "errors": errors,
    }
