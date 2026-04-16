/**
 * Browser-side structured extraction for large document files.
 *
 * Files larger than the Bedrock 4.5 MB per-document limit are extracted
 * in the browser and sent as a Markdown file instead. Markdown is much
 * more token-efficient than HTML (no tag overhead) and Claude parses it
 * natively, which lowers both latency and token cost.
 *
 * Output format: <baseName>_extracted.md (text/markdown)
 *
 * Per-format strategy:
 *   PDF   – pdfjs-dist with line/paragraph grouping and heading inference
 *   DOCX  – mammoth.convertToHtml() → compact Markdown transformation
 *   XLSX/XLS – SheetJS sheet_to_json() → Markdown pipe tables
 */

export const MAX_EXTRACTABLE_FILE_SIZE_MB = 50;
export const MAX_EXTRACTABLE_FILE_SIZE_BYTES =
  MAX_EXTRACTABLE_FILE_SIZE_MB * 1024 * 1024;

// Extracted Markdown is capped here to keep the resulting file well under 4.5 MB
const MAX_EXTRACTED_MARKDOWN_CHARS = 3_500_000;

export const EXTRACTABLE_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls'];

export function isExtractable(fileName: string): boolean {
  const lower = fileName.toLowerCase();
  return EXTRACTABLE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// ── Markdown utilities ────────────────────────────────────────────────────────

/**
 * Escape characters that carry special meaning in Markdown so that literal
 * document text doesn't accidentally render as formatting. Kept minimal —
 * over-escaping hurts readability for the model.
 */
function escapeMd(str: string): string {
  return str.replace(/([\\`*_[\]<>|])/g, '\\$1');
}

function collapseBlankLines(md: string): string {
  return md.replace(/\n{3,}/g, '\n\n').trim() + '\n';
}

// ── PDF ───────────────────────────────────────────────────────────────────────

interface PdfTextItem {
  str: string;
  transform: number[]; // [scaleX, skewX, skewY, scaleY, tx, ty] — tx=x, ty=y
  height: number;
}

async function extractPdfAsMarkdown(file: File): Promise<string> {
  const pdfjsLib = await import('pdfjs-dist');
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).href;

  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() })
    .promise;

  const parts: string[] = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();

    if (pageNum > 1) {
      // Thematic break between pages is the Markdown equivalent of <hr>
      parts.push('\n---\n');
    }

    // Filter to real text items with content
    const items = (textContent.items as PdfTextItem[]).filter(
      (item) => 'str' in item && item.str.trim() !== ''
    );

    if (items.length === 0) {continue;}

    // PDF y-axis is bottom-up; sort top-to-bottom, left-to-right
    items.sort((a, b) => {
      const dy = b.transform[5] - a.transform[5];
      if (Math.abs(dy) > 2) {return dy;}
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
        paragraphs[paragraphs.length - 1].lines.push(lineText);
      }
      prevY = lineY;
    }

    for (const para of paragraphs) {
      if (para.lines.length === 0) {continue;}
      if (para.isHeading) {
        parts.push(`\n## ${para.lines.join(' ').trim()}\n`);
      } else {
        // Join physical lines of a paragraph into a single wrapped paragraph
        parts.push(`${para.lines.join(' ').trim()}\n`);
      }
    }
  }

  return collapseBlankLines(parts.join('\n'));
}

// ── DOCX ──────────────────────────────────────────────────────────────────────

/**
 * Convert the constrained HTML subset that mammoth produces into Markdown.
 * mammoth emits a small, well-defined set of elements (h1–h6, p, ul/ol/li,
 * strong/em, a, br, table/tr/td/th, img), so a bespoke walker is sufficient
 * and avoids pulling in a general-purpose HTML-to-Markdown dependency.
 */
function htmlToMarkdown(html: string): string {
  const parser = new DOMParser();
  const doc = parser.parseFromString(
    `<!DOCTYPE html><html><body>${html}</body></html>`,
    'text/html'
  );
  const md = renderNode(doc.body, { listDepth: 0, inPre: false });
  return collapseBlankLines(md);
}

interface RenderCtx {
  listDepth: number;
  inPre: boolean;
}

function renderNode(node: Node, ctx: RenderCtx): string {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = node.textContent || '';
    return ctx.inPre ? text : escapeMd(text);
  }
  if (node.nodeType !== Node.ELEMENT_NODE) {return '';}

  const el = node as Element;
  const tag = el.tagName.toLowerCase();

  const renderChildren = (overrideCtx?: Partial<RenderCtx>) =>
    Array.from(el.childNodes)
      .map((c) => renderNode(c, { ...ctx, ...overrideCtx }))
      .join('');

  switch (tag) {
    case 'h1':
    case 'h2':
    case 'h3':
    case 'h4':
    case 'h5':
    case 'h6': {
      const level = Number(tag[1]);
      return `\n\n${'#'.repeat(level)} ${renderChildren().trim()}\n\n`;
    }
    case 'p':
      return `\n\n${renderChildren().trim()}\n\n`;
    case 'br':
      return '  \n';
    case 'strong':
    case 'b': {
      const inner = renderChildren().trim();
      return inner ? `**${inner}**` : '';
    }
    case 'em':
    case 'i': {
      const inner = renderChildren().trim();
      return inner ? `*${inner}*` : '';
    }
    case 'u':
      return renderChildren(); // Markdown has no underline — render plain
    case 'a': {
      const href = el.getAttribute('href') || '';
      const inner = renderChildren().trim();
      if (!inner) {return '';}
      return href ? `[${inner}](${href})` : inner;
    }
    case 'ul':
    case 'ol': {
      const items = Array.from(el.children).filter(
        (c) => c.tagName.toLowerCase() === 'li'
      );
      const marker = (i: number) => (tag === 'ol' ? `${i + 1}.` : '-');
      const indent = '  '.repeat(ctx.listDepth);
      const lines = items.map((li, i) => {
        const content = renderNode(li, {
          ...ctx,
          listDepth: ctx.listDepth + 1,
        })
          .trim()
          .replace(/\n+/g, ' ');
        return `${indent}${marker(i)} ${content}`;
      });
      return `\n${lines.join('\n')}\n`;
    }
    case 'li':
      return renderChildren();
    case 'table':
      return renderTable(el);
    case 'img': {
      // Drop images — they're typically base64 data URIs in mammoth output,
      // which would balloon the token count. Replace with a placeholder.
      const alt = el.getAttribute('alt') || 'image';
      return `*[${alt}]*`;
    }
    case 'code':
      return `\`${el.textContent || ''}\``;
    case 'pre':
      return `\n\n\`\`\`\n${el.textContent || ''}\n\`\`\`\n\n`;
    case 'blockquote':
      return renderChildren()
        .split('\n')
        .map((l) => (l.trim() ? `> ${l}` : l))
        .join('\n');
    case 'script':
    case 'style':
      return '';
    default:
      return renderChildren();
  }
}

function renderTable(table: Element): string {
  const rows = Array.from(table.querySelectorAll('tr'));
  if (rows.length === 0) {return '';}

  const cellText = (cell: Element) =>
    (cell.textContent || '')
      .replace(/\s+/g, ' ')
      .replace(/\|/g, '\\|')
      .trim();

  const matrix: string[][] = rows.map((r) =>
    Array.from(r.children)
      .filter((c) => ['td', 'th'].includes(c.tagName.toLowerCase()))
      .map(cellText)
  );

  const colCount = Math.max(...matrix.map((r) => r.length), 0);
  if (colCount === 0) {return '';}

  // Pad to uniform column count
  for (const row of matrix) {
    while (row.length < colCount) {row.push('');}
  }

  const header = matrix[0];
  const body = matrix.slice(1);
  const sep = Array(colCount).fill('---');

  const fmt = (row: string[]) => `| ${row.join(' | ')} |`;
  const lines = [fmt(header), fmt(sep), ...body.map(fmt)];

  return `\n\n${lines.join('\n')}\n\n`;
}

async function extractDocxAsMarkdown(file: File): Promise<string> {
  const mammoth = await import('mammoth');
  const result = await mammoth.convertToHtml({
    arrayBuffer: await file.arrayBuffer(),
  });
  return htmlToMarkdown(result.value);
}

// ── XLSX / XLS ────────────────────────────────────────────────────────────────

async function extractSpreadsheetAsMarkdown(file: File): Promise<string> {
  const XLSX = await import('xlsx');
  const workbook = XLSX.read(await file.arrayBuffer(), { type: 'buffer' });

  const sections: string[] = [];

  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];

    // Extract as 2D array of strings
    const rows: unknown[][] = XLSX.utils.sheet_to_json(sheet, {
      header: 1,
      blankrows: false,
      defval: '',
      raw: false,
    });

    sections.push(`## ${sheetName}\n`);

    if (rows.length === 0) {
      sections.push('_No data._\n');
      continue;
    }

    const stringRows = rows.map((r) =>
      r.map((v) =>
        String(v ?? '')
          .replace(/\s+/g, ' ')
          .replace(/\|/g, '\\|')
          .trim()
      )
    );

    const colCount = Math.max(...stringRows.map((r) => r.length), 0);
    if (colCount === 0) {
      sections.push('_No data._\n');
      continue;
    }

    // Normalise each row to colCount cells
    for (const r of stringRows) {
      while (r.length < colCount) {r.push('');}
    }

    const header = stringRows[0];
    const body = stringRows.slice(1);
    const sep = Array(colCount).fill('---');
    const fmt = (row: string[]) => `| ${row.join(' | ')} |`;

    sections.push([fmt(header), fmt(sep), ...body.map(fmt)].join('\n'));
    sections.push('');
  }

  return collapseBlankLines(sections.join('\n'));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Extract structured Markdown from a large document and return it as a new File.
 *
 * The returned file has an `_extracted.md` suffix. Markdown preserves document
 * structure (headings, paragraphs, lists, tables) at a fraction of the token
 * cost of HTML, which reduces latency and cost when the model processes it.
 *
 * Throws if the file type is unsupported or nothing could be extracted.
 */
export async function extractTextFromFile(file: File): Promise<File> {
  const lower = file.name.toLowerCase();

  let markdown: string;

  if (lower.endsWith('.pdf')) {
    markdown = await extractPdfAsMarkdown(file);
  } else if (lower.endsWith('.docx')) {
    markdown = await extractDocxAsMarkdown(file);
  } else if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
    markdown = await extractSpreadsheetAsMarkdown(file);
  } else {
    throw new Error('Text extraction is not supported for this file type.');
  }

  if (!markdown.trim()) {
    throw new Error('No content could be extracted from this file.');
  }

  if (markdown.length > MAX_EXTRACTED_MARKDOWN_CHARS) {
    markdown =
      markdown.slice(0, MAX_EXTRACTED_MARKDOWN_CHARS) +
      '\n\n<!-- Content truncated: file too large to include in full -->\n';
  }

  const blob = new Blob([markdown], { type: 'text/markdown' });
  const baseName = file.name.replace(/\.[^.]+$/, '');
  return new File([blob], `${baseName}_extracted.md`, {
    type: 'text/markdown',
  });
}

/** Format bytes to a human-readable string (KB or MB). */
export function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${Math.round(bytes / 1024)} KB`;
}
