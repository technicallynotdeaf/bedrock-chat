# Bedrock Chat — Claude Code Session Memory

## Project Overview
AWS-native chatbot using Amazon Bedrock. Deployed via CDK + CodeBuild from this repo (DPenniket/bedrock-chat, branch `v4`).
- **Frontend**: React/Vite, hosted on S3/CloudFront
- **Backend**: Lambda (Python 3.13), API Gateway (WebSocket + REST)
- **Repo**: https://github.com/DPenniket/bedrock-chat

## Deploy Command (run from CloudShell)
```bash
./bin.sh --disable-self-register --bedrock-region ap-southeast-2 --repo-url https://github.com/DPenniket/bedrock-chat.git --version v4
```
- CodeBuild does: `git clone --branch v4` then `npx cdk deploy --require-approval never --all`
- Deploy is to `ap-southeast-2` (Sydney)

---

## Data Sovereignty Requirement (STRICT — DO NOT VIOLATE)

**All inference must remain within Australia at all times.** Data must never leave Australian soil.

- AU inference profiles must only include `ap-southeast-2` (Sydney) and `ap-southeast-4` (Melbourne)
- **Never add `ap-southeast-6` (Auckland, New Zealand)** — Auckland is outside Australia and adding it would violate data sovereignty requirements
- `enableBedrockGlobalInference` must remain `false` — global profiles route outside Australia
- Do not add any non-Australian regions (e.g. `us-*`, `eu-*`, `ap-northeast-*`, `ap-southeast-6`) to the AU inference profile mappings

---

## Critical Lessons Learned (DO NOT REPEAT THESE MISTAKES)

### 1. Never change the WebSocket chunk size from 32KB
API Gateway WebSocket has a **hard 32KB per-message limit**. Previous attempts to increase to 100KB or 128KB broke everything. The 32KB chunk size is proven to work. Do not change it.

### 2. Frontend and backend must stay in sync
Changing the backend START response format (from `"Session started."` to JSON `{"uploadUrl": "..."}`) without ensuring backward compatibility in the frontend broke ALL chat — not just large documents. Always handle both old and new response formats.

### 3. The "Message part received." ack format is sacred
Changing the ack format (e.g., to `{"ack": index}`) caused frontend/backend mismatches. The v3 format is `"Message part received."` — a plain string. Do not change this.

### 4. Document chunking threshold must be 2.5MB
- Documents up to ~2.5MB pass through Bedrock natively without issues
- Setting the threshold too low (50KB, 500KB) broke normal document handling
- Setting it too high (4.5MB) caused context window overflow and API Gateway timeouts
- **2.5MB is the tested sweet spot** — `LARGE_DOCUMENT_THRESHOLD_BYTES = 2_500_000`

### 5. S3 direct upload from browser requires CORS on the S3 bucket
Pre-signed URLs for S3 PUTs from the browser require CORS configuration on the bucket. Without it, the browser blocks the request. CORS must be deployed via CDK before the S3 direct upload path works. The sequential chunking fallback exists for when CORS isn't deployed yet.

### 6. The bot knowledge upload already has a working S3 pre-signed URL flow
`GET /bot/{bot_id}/presigned-url` → S3 upload via `axios.put()`. The document bucket has CORS configured in `cdk/lib/bedrock-chat-stack.ts` (lines 290-299). This pattern works and can be referenced for future S3 upload features.

### 7. Test with ALL document sizes, not just large ones
Multiple changes that "fixed" large documents broke small documents or normal chat. Always test: no attachment, small attachment (<100KB), medium (1-2MB), and large (>2.5MB).

---

## Current Architecture: Large Document Upload (branch: `claude/add-document-chunking-6ZRhB`)

### Two-layer approach:
1. **WebSocket delivery layer** — gets the payload from browser to Lambda
2. **Document chunking layer** — manages context window for Bedrock

These are independent. The delivery layer doesn't know about documents; the chunking layer doesn't know about WebSocket.

### WebSocket Delivery Protocol

**Frontend** (`frontend/src/hooks/usePostMessageStreaming.ts`):
1. **START**: Client sends `{step: "START", token}`
2. Backend returns either:
   - `{"uploadUrl": "..."}` (new backend) → client tries S3 direct upload via XHR
   - `"Session started."` (old backend) → client falls back to sequential chunking
3. **If S3 upload succeeds**: Client sends END immediately
4. **If S3 upload fails (CORS)**: Falls back to sequential WebSocket chunking:
   - Sends BODY chunks ONE AT A TIME (32KB each), waits for `"Message part received."` ack before sending next
   - Sequential sending avoids Lambda throttling (only 1 concurrent invocation)
   - Progress bar updates on each ack: `uploadProgress = chunkAckCount / totalChunks * 100`
5. **END**: After all chunks acked (or S3 upload succeeded), sends `{step: "END", token}`

**Backend** (`backend/app/websocket.py`):
- **START**: Verifies token, stores session in S3, generates pre-signed PUT URL, returns it as JSON body
- **BODY**: Stores chunk to S3 at `ws-chunks/{connection_id}/{index:010d}`, returns `"Message part received."`
- **END**: Tries reading direct S3 upload (`ws-chunks/{connection_id}/payload`) first; falls back to chunk assembly if not found. Then processes via `process_chat_input()` → Bedrock

**CDK** (`cdk/lib/constructs/websocket.ts`):
- S3 bucket `largePayloadSupportBucket` has CORS configured for PUT from `*` origin
- 1-day lifecycle rule on `ws-chunks/` prefix for cleanup
- Lambda: 512MB memory, 15-minute timeout

### Document Chunking Layer

**Backend** (`backend/app/document_chunker.py`):
- Threshold: `LARGE_DOCUMENT_THRESHOLD_BYTES = 2_500_000` (2.5MB)
- Documents >2.5MB → text extraction → truncate to 320K chars (~80K tokens) → split into 50K char chunks
- Supports: PDF (pypdf), DOCX (python-docx), XLSX (openpyxl), plain text
- If text extraction fails for PDFs: tries splitting to first 100 pages, or returns original if <4.5MB
- Called in `backend/app/usecases/chat.py` via `process_attachments_for_context_window()`

### Upload Progress Bar

**Frontend** (`frontend/src/components/ChatMessage.tsx`):
- `uploadProgress` state in `usePostMessageStreaming` Zustand store (0-100 or null)
- S3 direct path: real-time byte progress via `XMLHttpRequest.upload.onprogress`
- Sequential chunk path: updates on each chunk ack
- Displayed as a blue progress bar with percentage in the assistant message area
- Replaces typing indicator during upload; typing indicator takes over after upload completes

---

## Existing Error Handling

All Lambda paths return `statusCode: 200`. Actual errors sent via `post_to_connection`:
- START invalid token → `notificator.notify(ERROR)` + return 200
- BODY invalid index → return 200 with "Error." body
- END/processing errors → `notificator.notify(ERROR)` with reason + return 200
- Outer handler except → `notificator.notify(ERROR)` with reason + return 200

Frontend handles:
- `"Error."` and `"Internal server error"` during chunking → retry chunk
- API Gateway JSON errors during chunking → retry chunk
- `{"status": "ERROR", "reason": "..."}` → display reason in UI
- `{"message": "Endpoint request timed out"}` → filtered/ignored (Lambda continues via post_to_connection)

---

## Known Remaining Issues

### "Request failed with status code 500" in app
User reports seeing this error in the app even when things appear to work. Origin not yet identified — likely a REST API call (possibly `GET /bot?kind=private`) failing but not blocking the main chat flow.

### Document upload tested working up to 4.3MB
The sequential chunking fallback + document chunker combination is confirmed working for documents up to at least 4.3MB. The 4.5MB Bedrock per-document limit is the hard ceiling.

---

## File Locations
- Backend WebSocket handler: `backend/app/websocket.py`
- Backend document chunker: `backend/app/document_chunker.py`
- Backend chat processing: `backend/app/usecases/chat.py`
- Backend document chunker tests: `backend/tests/test_document_chunker.py`
- Frontend streaming hook: `frontend/src/hooks/usePostMessageStreaming.ts`
- Frontend chat message (progress bar): `frontend/src/components/ChatMessage.tsx`
- CDK WebSocket construct: `cdk/lib/constructs/websocket.ts`
- CDK main stack (document bucket CORS): `cdk/lib/bedrock-chat-stack.ts`
- S3 presigned URL utility: `backend/app/utils.py` (lines 73-96)
- Bot presigned URL endpoint: `backend/app/routes/bot.py` → `issue_presigned_url()`
- Frontend bot upload: `frontend/src/hooks/useBotApi.ts` → `uploadFile()`
- Supported file extensions: `frontend/src/constants/supportedAttachedFiles.ts`
- Deploy script: `bin.sh` + `deploy.yml`
- Dev branch: `claude/add-document-chunking-6ZRhB`

## Attachment Data Flow (for reference)
```
Browser: File → ArrayBuffer → base64 string → PostMessageRequest.message.content[].body
WebSocket: JSON payload (includes base64) → chunked 32KB → Lambda → S3 → reassemble
Backend: ChatInput → Base64EncodedBytes (auto-decodes to bytes) → AttachmentContentModel
Chunker: If >2.5MB → extract text → truncate to 320K chars → TextContentModel blocks
Bedrock: Either raw document bytes (native) or extracted text chunks
```

---

## Latency Optimizations Applied (branch: `claude/optimize-bot-latency-h4i4q`)

### Tier 1 — Backend request-path optimizations

1. **Cached boto3 Bedrock clients** (`backend/app/utils.py`)
   - All `get_bedrock_*_client()` functions now return cached clients keyed by `(service, region)`
   - Configured with `adaptive` retry mode and `max_pool_connections=10`
   - Saves ~50-150ms per Bedrock call on warm Lambda invocations

2. **Batched WebSocket token streaming** (`backend/app/websocket.py`)
   - `NotificationSender` now buffers streaming tokens and flushes every 50ms or 5 tokens
   - Reduces `post_to_connection` API calls from ~500 to ~100 per typical response
   - Applies to both STREAMING and REASONING tokens; flushes on status change
   - New command type `stream_token` alongside existing `notify` and `finish`

3. **Removed redundant S3 session read on END step** (`backend/app/websocket.py`)
   - The END handler was reading session.json from S3 just for `user_id`, but the JWT token already provides it via `verify_token` → `User.from_decoded_token`

4. **Deferred S3 chunk cleanup** (`backend/app/websocket.py`)
   - `_cleanup_s3_chunks()` now runs in a background daemon thread *after* `process_chat_input` returns
   - The 1-day lifecycle rule on `ws-chunks/` prefix handles anything missed

5. **Reduced Bedrock throttle retry delay** (`backend/app/stream.py`, `backend/app/bedrock.py`)
   - Changed `@retry` delay from 60s to 2s (with backoff=2, jitter=(0,2))
   - Previous 60s delay was catastrophic for user experience on throttling

### Tier 2 — I/O parallelization and import deferral

6. **Parallel web URL fetches** (`backend/app/web_url_handler.py`)
   - `fetch_urls_content()` now uses `ThreadPoolExecutor` to fetch up to 5 URLs concurrently
   - Previously sequential: 5 URLs × 15s timeout = up to 75s → now ~15s max

7. **Parallel PDF URL downloads** (`backend/app/usecases/chat.py`)
   - `process_pdf_urls_in_message()` now downloads PDFs in parallel with `ThreadPoolExecutor`

8. **Concurrent PDF + web URL processing** (`backend/app/usecases/chat.py`)
   - `process_pdf_urls_in_message` and `process_web_urls_in_message` now run concurrently in the `chat()` function using a 2-worker ThreadPoolExecutor

9. **Lazy imports in WebSocket handler** (`backend/app/websocket.py`)
   - Heavy imports (`app.usecases.chat`, `app.stream`, `app.agents`, `app.repositories.conversation`, `app.routes.schemas.conversation`) are deferred to the END step
   - Module-level imports reduced to: `json`, `logging`, `os`, `time`, `boto3`, `app.auth`, `app.user`
   - Type hints preserved via `TYPE_CHECKING` + `from __future__ import annotations`
   - Reduces cold start time for START and BODY steps

### Critical: Do NOT change these values
- `_STREAM_BATCH_INTERVAL = 0.05` (50ms) — lower causes excessive API calls, higher causes visible lag
- `_STREAM_BATCH_MAX_TOKENS = 5` — balanced between batching efficiency and streaming feel
- Throttle retry `delay=2` — must be short for UX but non-zero to avoid hammering Bedrock
