// Text handling for the in-browser detector.
// A line-by-line port of prepare_hape.clean_text and detect.sentence_chunks,
// so the browser feeds the model exactly what the Python version does.
(function (root) {
  "use strict";

  const CHUNK_TOKENS = 256;
  const MIN_LAST_CHUNK = 128;

  const QUOTE_MAP = { "‘": "'", "’": "'", "“": '"', "”": '"',
                      "–": "-", "—": "-", "…": "...", " ": " " };
  const QUOTE_RE = /[‘’“”–—… ]/g;
  const COCA_REDACT = /(?:@\s*){2,}|\s@\s/g;
  const HTML_TAG = /<\/?(p|br|h\d)\s*\/?>/gi;
  const LEAD_IN = new RegExp(
    "^\\s*(?:sure[,!.]?\\s*)?(?:here(?:'s| is| are)\\b[^:\\n]{0,150}?" +
    "(?:continu|next|complet|rest of|remain|style|tone|words)[^:\\n]{0,100}:" +
    "|continuation\\s*:)\\s*", "i");
  const SENT_END = /[.!?]["')\]]?\s+(?=["'(\[]?[A-Z])/;
  const LAST_END = /^([\s\S]*[.!?]["')\]]?)(?:\s|$)/;
  const SPLIT_SENT = /(?:(?<=[.!?])|(?<=[.!?]["')\]]))\s+(?=["'(\[]?[A-Z0-9])/;

  function trimPartialSentences(text) {
    if (/^(\.\.\.|[a-z])/.test(text)) {
      const m = SENT_END.exec(text);
      if (m) text = text.slice(m.index + m[0].length);
    }
    const m = LAST_END.exec(text);
    return m ? m[1] : text;
  }

  function cleanText(text) {
    text = String(text).normalize("NFKC");
    text = text.replace(/\s*(?:--|—|–)\s*/g, " - ");
    text = text.replace(QUOTE_RE, (c) => QUOTE_MAP[c]);
    text = text.replace(HTML_TAG, " ");
    text = text.replace(COCA_REDACT, " ");
    text = text.replace(/\s+/g, " ").trim();
    text = text.replace(LEAD_IN, "");
    text = text.replace(/([,;:])\s*\.\s/g, "$1 ");
    text = text.replace(/\s\.\s(?=[a-z])/g, " ");
    text = text.replace(/ ([.,!?;:])/g, "$1");
    text = trimPartialSentences(text);
    return text.trim();
  }

  // returns [{ids, text}] - the token ids fed to the model and the text they cover
  function sentenceChunks(text, enc) {
    const encode = (s) => enc.encode(s, [], []);
    const sentences = text.split(SPLIT_SENT).filter((s) => s);
    const chunks = [];
    let cur = [], curText = [];
    for (const s of sentences) {
      const ids = encode(cur.length ? " " + s : s);
      if (cur.length && cur.length + ids.length > CHUNK_TOKENS) {
        chunks.push({ ids: cur, text: curText.join(" ") });
        cur = encode(s);
        curText = [s];
      } else {
        cur = cur.concat(ids);
        curText.push(s);
      }
    }
    if (cur.length) chunks.push({ ids: cur, text: curText.join(" ") });
    for (const c of chunks) {
      if (c.ids.length > CHUNK_TOKENS) { c.ids = c.ids.slice(0, CHUNK_TOKENS); c.truncated = true; }
    }
    if (chunks.length > 1 && chunks[chunks.length - 1].ids.length < MIN_LAST_CHUNK) {
      chunks[chunks.length - 1].skipped = true;   // too short to score on its own
    }
    return chunks;
  }

  root.Detector = { cleanText, sentenceChunks, CHUNK_TOKENS, MIN_LAST_CHUNK };
  if (typeof module !== "undefined") module.exports = root.Detector;
})(typeof window !== "undefined" ? window : globalThis);
