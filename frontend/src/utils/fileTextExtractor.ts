/**
 * Browser-side text extraction for large document files.
 *
 * Files larger than the Bedrock 4.5 MB per-document limit are text-extracted
 * in the browser and sent as a smaller plain-text file instead.
 */

// Maximum file size we will attempt to extract (files larger than this are rejected)
export const MAX_EXTRACTABLE_FILE_SIZE_MB = 50;
export const MAX_EXTRACTABLE_FILE_SIZE_BYTES =
  MAX_EXTRACTABLE_FILE_SIZE_MB * 1024 * 1024;

// Extracted text is capped here to keep the resulting .txt well under 4.5 MB
const MAX_EXTRACTED_TEXT_CHARS = 3_500_000;

// File extensions that support browser-side text extraction
export const EXTRACTABLE_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls'];

export function isExtractable(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return EXTRACTABLE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// ── PDF ──────────────────────────────────────────────────────────────────────

async function extractPdfText(file: File): Promise<string> {
  const pdfjsLib = await import('pdfjs-dist');

  // Vite turns `new URL(specifier, import.meta.url)` into a static asset URL
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).href;

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  const pages: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      .map((item) => ('str' in item ? (item as { str: string }).str : ''))
      .join(' ');
    pages.push(pageText);
  }

  return pages.join('\n\n');
}

// ── DOCX ─────────────────────────────────────────────────────────────────────

async function extractDocxText(file: File): Promise<string> {
  const mammoth = await import('mammoth');
  const arrayBuffer = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer });
  return result.value;
}

// ── XLSX / XLS ────────────────────────────────────────────────────────────────

async function extractSpreadsheetText(file: File): Promise<string> {
  const XLSX = await import('xlsx');
  const arrayBuffer = await file.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer, { type: 'buffer' });

  const sections: string[] = [];
  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    const csv = XLSX.utils.sheet_to_csv(sheet);
    sections.push(`=== Sheet: ${sheetName} ===\n${csv}`);
  }

  return sections.join('\n\n');
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Extract plain text from a large document file and return it as a new File.
 *
 * The returned file has a `.txt` extension and is typically much smaller than
 * the original, making it suitable for submission via the Bedrock Converse API.
 *
 * Throws if the file type is not supported or no text could be extracted.
 */
export async function extractTextFromFile(file: File): Promise<File> {
  const lower = file.name.toLowerCase();

  let text: string;

  if (lower.endsWith('.pdf')) {
    text = await extractPdfText(file);
  } else if (lower.endsWith('.docx')) {
    text = await extractDocxText(file);
  } else if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
    text = await extractSpreadsheetText(file);
  } else {
    throw new Error(`Text extraction is not supported for this file type.`);
  }

  if (!text.trim()) {
    throw new Error('No text could be extracted from this file.');
  }

  if (text.length > MAX_EXTRACTED_TEXT_CHARS) {
    text =
      text.slice(0, MAX_EXTRACTED_TEXT_CHARS) +
      '\n\n[Content truncated: file too large to include in full]';
  }

  const blob = new Blob([text], { type: 'text/plain' });
  const baseName = file.name.replace(/\.[^.]+$/, '');
  return new File([blob], `${baseName}_extracted.txt`, { type: 'text/plain' });
}

/** Format bytes to a human-readable string (KB or MB). */
export function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${Math.round(bytes / 1024)} KB`;
}
