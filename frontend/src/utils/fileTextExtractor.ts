/**
 * Browser-side structured extraction for large document files.
 *
 * Files larger than the Bedrock 4.5 MB per-document limit are extracted
 * in the browser and sent as a semantic HTML file instead, which Claude
 * can parse far better than flat plain text.
 *
 * Output format: <baseName>_extracted.html (text/html)
 *
 * Per-format strategy:
 *   PDF   – pdfjs-dist with line/paragraph grouping and heading inference
 *   DOCX  – mammoth.convertToHtml() preserves headings, lists, tables
 *   XLSX/XLS – SheetJS sheet_to_html() produces one HTML table per sheet
 */

export const MAX_EXTRACTABLE_FILE_SIZE_MB = 50;
export const MAX_EXTRACTABLE_FILE_SIZE_BYTES =
  MAX_EXTRACTABLE_FILE_SIZE_MB * 1024 * 1024;

// Extracted HTML is capped here to keep the resulting file well under 4.5 MB
const MAX_EXTRACTED_HTML_CHARS = 3_500_000;

export const EXTRACTABLE_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls'];

export function isExtractable(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return EXTRACTABLE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function wrapHtml(body: string, title: string): string {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title></head><body>\n${body}\n</body></html>`;
}

// ── PDF ───────────────────────────────────────────────────────────────────────

interface PdfTextItem {
  str: string;
  transform: number[]; // [scaleX, skewX, skewY, scaleY, tx, ty] — tx=x, ty=y
  height: number;
}

async function extractPdfAsHtml(file: File): Promise<string> {
  const pdfjsLib = await import('pdfjs-dist');
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).href;

  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() })
    .promise;

  const bodyParts: string[] = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();

    if (pageNum > 1) {
      bodyParts.push('<hr>');
    }

    // Filter to real text items with content
    const items = (textContent.items as PdfTextItem[]).filter(
      (item) => 'str' in item && item.str.trim() !== ''
    );

    if (items.length === 0) continue;

    // PDF y-axis is bottom-up; sort top-to-bottom, left-to-right
    items.sort((a, b) => {
      const dy = b.transform[5] - a.transform[5];
      if (Math.abs(dy) > 2) return dy;
      return a.transform[4] - b.transform[4];
    });

    // Group items into visual lines by y proximity
    const lines: PdfTextItem[][] = [];
    let currentLine: PdfTextItem[] = [items[0]];
    for (let i = 1; i < items.length; i++) {
      const dy = Math.abs(
        items[i].transform[5] - currentLine[0].transform[5]
      );
      if (dy <= 3) {
        currentLine.push(items[i]);
      } else {
        lines.push(currentLine);
        currentLine = [items[i]];
      }
    }
    lines.push(currentLine);

    // Determine median body font height for heading / paragraph-gap heuristics
    const heights = items
      .map((i) => i.height)
      .filter((h) => h > 0)
      .sort((a, b) => a - b);
    const medianH = heights[Math.floor(heights.length / 2)] || 10;

    // Group lines into paragraphs by vertical gap
    const paragraphs: { lines: string[]; isHeading: boolean }[] = [
      { lines: [], isHeading: false },
    ];
    let prevY = lines[0][0].transform[5];

    for (const line of lines) {
      const lineY = line[0].transform[5];
      const gap = prevY - lineY;
      const avgH =
        line.reduce((s, it) => s + (it.height || medianH), 0) / line.length;
      const isHeading = avgH > medianH * 1.4;

      // Start a new paragraph on a large gap, or when toggling heading/body
      if (
        (gap > medianH * 1.8 ||
          isHeading !== paragraphs[paragraphs.length - 1].isHeading) &&
        paragraphs[paragraphs.length - 1].lines.length > 0
      ) {
        paragraphs.push({ lines: [], isHeading });
      } else if (paragraphs.length === 1 && paragraphs[0].lines.length === 0) {
        paragraphs[0].isHeading = isHeading;
      }

      const lineText = line.map((it) => it.str).join(' ').trim();
      if (lineText) {
        paragraphs[paragraphs.length - 1].lines.push(escapeHtml(lineText));
      }
      prevY = lineY;
    }

    for (const para of paragraphs) {
      if (para.lines.length === 0) continue;
      const content = para.lines.join('<br>');
      bodyParts.push(
        para.isHeading ? `<h2>${content}</h2>` : `<p>${content}</p>`
      );
    }
  }

  return wrapHtml(bodyParts.join('\n'), file.name);
}

// ── DOCX ──────────────────────────────────────────────────────────────────────

async function extractDocxAsHtml(file: File): Promise<string> {
  const mammoth = await import('mammoth');
  const result = await mammoth.convertToHtml({
    arrayBuffer: await file.arrayBuffer(),
  });
  return wrapHtml(result.value, file.name);
}

// ── XLSX / XLS ────────────────────────────────────────────────────────────────

async function extractSpreadsheetAsHtml(file: File): Promise<string> {
  const XLSX = await import('xlsx');
  const workbook = XLSX.read(await file.arrayBuffer(), { type: 'buffer' });

  const sections: string[] = [];

  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];

    // sheet_to_html returns a full HTML document; extract just the <table>
    const fullHtml = XLSX.utils.sheet_to_html(sheet);
    const tableMatch = fullHtml.match(/<table[\s\S]*?<\/table>/i);
    const table = tableMatch
      ? tableMatch[0]
      : `<p><em>No data in sheet "${escapeHtml(sheetName)}"</em></p>`;

    sections.push(
      `<section>\n<h2>${escapeHtml(sheetName)}</h2>\n${table}\n</section>`
    );
  }

  return wrapHtml(sections.join('\n'), file.name);
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Extract structured HTML from a large document and return it as a new File.
 *
 * The returned file has an `_extracted.html` suffix. The HTML preserves as
 * much document structure as possible so that Claude can interpret it
 * accurately (headings, paragraphs, lists, tables, bold text).
 *
 * Throws if the file type is unsupported or nothing could be extracted.
 */
export async function extractTextFromFile(file: File): Promise<File> {
  const lower = file.name.toLowerCase();

  let html: string;

  if (lower.endsWith('.pdf')) {
    html = await extractPdfAsHtml(file);
  } else if (lower.endsWith('.docx')) {
    html = await extractDocxAsHtml(file);
  } else if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
    html = await extractSpreadsheetAsHtml(file);
  } else {
    throw new Error('Text extraction is not supported for this file type.');
  }

  if (!html.replace(/<[^>]*>/g, '').trim()) {
    throw new Error('No content could be extracted from this file.');
  }

  if (html.length > MAX_EXTRACTED_HTML_CHARS) {
    // Truncate body content, keeping valid HTML structure
    const truncated = html.slice(0, MAX_EXTRACTED_HTML_CHARS);
    html =
      truncated + '\n<!-- Content truncated: file too large to include in full -->\n</body></html>';
  }

  const blob = new Blob([html], { type: 'text/html' });
  const baseName = file.name.replace(/\.[^.]+$/, '');
  return new File([blob], `${baseName}_extracted.html`, { type: 'text/html' });
}

/** Format bytes to a human-readable string (KB or MB). */
export function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${Math.round(bytes / 1024)} KB`;
}
