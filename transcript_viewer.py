"""
Transcript Viewer — Video + SRT Review Tool

A local web app to review transcripts alongside video playback.
Click a transcript row to jump the video to that timestamp.
Browse all videos by transcript summary in the Library view.

Usage:
    python transcript_viewer.py --video-root C:\\Users\\k2\\videos-daytime --transcript-root C:\\Users\\k2\\videos-daytime_audio_large
    python transcript_viewer.py  (uses defaults)

Then open http://localhost:8888 in your browser.
"""

import os
import sys
import json
import argparse
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

# Defaults
DEFAULT_VIDEO_ROOT = r"C:\Users\k2\videos-daytime"
DEFAULT_TRANSCRIPT_ROOT = r"C:\Users\k2\videos-daytime_audio_large"
DEFAULT_PORT = 8888

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
TRANSCRIPT_EXTENSIONS = {'.srt'}


# ============================================================
# HTML Template (embedded)
# ============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcript Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d27;
    --bg-card: #21242f;
    --bg-hover: #2a2e3b;
    --bg-active: #1e3a5f;
    --text-primary: #e4e6ec;
    --text-secondary: #8b8fa3;
    --text-muted: #5c6070;
    --accent: #4f8ff7;
    --accent-glow: rgba(79, 143, 247, 0.15);
    --accent-dim: #3a6bc5;
    --highlight: #fbbf24;
    --highlight-bg: rgba(251, 191, 36, 0.08);
    --border: #2a2e3b;
    --success: #34d399;
    --danger: #f87171;
    --radius: 10px;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
}

/* ---- Top Bar ---- */
.top-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    z-index: 10;
}
.top-bar h1 {
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    white-space: nowrap;
    letter-spacing: -0.3px;
    cursor: pointer;
}
.top-bar h1:hover { opacity: 0.8; }
.top-bar select, .top-bar .nav-btn {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 12px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    min-width: 100px;
    transition: all 0.15s;
}
.top-bar select { min-width: 180px; }
.top-bar select:hover, .top-bar .nav-btn:hover { border-color: var(--accent-dim); }
.top-bar select:focus, .top-bar .nav-btn:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
.top-bar .nav-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.top-bar .spacer { flex: 1; }
.top-bar .info {
    font-size: 12px;
    color: var(--text-muted);
}

/* ---- Views ---- */
.view { display: none; height: calc(100vh - 52px); }
.view.active { display: flex; }

/* ---- Library View ---- */
.library-view {
    flex-direction: column;
    overflow: hidden;
}
.library-toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}
.library-toolbar .search-box {
    flex: 1;
    max-width: 500px;
}
.library-toolbar .count {
    font-size: 12px;
    color: var(--text-muted);
}
.library-toolbar select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-family: inherit;
    font-size: 13px;
}
.library-grid {
    flex: 1;
    overflow-y: auto;
    padding: 12px 20px;
}
.library-grid::-webkit-scrollbar { width: 6px; }
.library-grid::-webkit-scrollbar-track { background: transparent; }
.library-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.lib-row {
    display: flex;
    gap: 16px;
    padding: 12px 16px;
    margin-bottom: 4px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.12s;
}
.lib-row:hover {
    background: var(--bg-hover);
    border-color: var(--border);
}
.lib-row .lib-date {
    font-size: 12px;
    color: var(--accent-dim);
    font-weight: 500;
    min-width: 120px;
    padding-top: 2px;
}
.lib-row .lib-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    min-width: 300px;
}
.lib-row .lib-summary {
    font-size: 13px;
    color: var(--text-secondary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.lib-row .lib-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    white-space: nowrap;
    align-self: center;
}
.lib-row .lib-badge.has-video {
    background: rgba(52, 211, 153, 0.12);
    color: var(--success);
}
.lib-row .lib-badge.no-video {
    background: rgba(248, 113, 113, 0.12);
    color: var(--danger);
}

/* ---- Player View ---- */
.player-view { display: none; }
.player-view.active { display: flex; }

/* ---- Video Panel ---- */
.video-panel {
    flex: 0 0 55%;
    display: flex;
    flex-direction: column;
    background: #000;
    position: relative;
}
.video-panel video {
    width: 100%;
    flex: 1;
    object-fit: contain;
    background: #000;
}
.video-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-secondary);
}
.video-controls button {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: all 0.15s;
}
.video-controls button:hover { background: var(--bg-hover); border-color: var(--accent-dim); }
.video-controls button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.video-controls .time-display {
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    color: var(--text-primary);
    font-weight: 500;
    min-width: 120px;
}
.speed-control select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    font-family: inherit;
}

/* ---- Transcript Panel ---- */
.transcript-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    border-left: 1px solid var(--border);
}
.transcript-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}
.transcript-header h2 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.transcript-header .spacer { flex: 1; }
.search-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: inherit;
    width: 200px;
    transition: border-color 0.15s;
}
.search-box:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow); }
.search-box::placeholder { color: var(--text-muted); }

.transcript-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
    scroll-behavior: smooth;
}
.transcript-list::-webkit-scrollbar { width: 6px; }
.transcript-list::-webkit-scrollbar-track { background: transparent; }
.transcript-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.transcript-list::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

.seg-row {
    display: flex;
    gap: 12px;
    padding: 8px 16px;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: all 0.12s;
}
.seg-row:hover {
    background: var(--bg-hover);
}
.seg-row.active {
    background: var(--bg-active);
    border-left-color: var(--accent);
}
.seg-row.active .seg-time { color: var(--accent); }
.seg-row .seg-time {
    font-size: 12px;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    min-width: 80px;
    padding-top: 1px;
    font-weight: 500;
}
.seg-row .seg-text {
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-primary);
}
.seg-row.hidden { display: none; }
mark {
    background: var(--highlight);
    color: #000;
    border-radius: 2px;
    padding: 0 2px;
}

/* ---- Empty State ---- */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    gap: 12px;
}
.empty-state .icon { font-size: 48px; opacity: 0.3; }
.empty-state p { font-size: 14px; }

/* ---- Keyboard Hint ---- */
.kbd-hints {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
}
.kbd-hints kbd {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    font-family: inherit;
    font-size: 11px;
    color: var(--text-secondary);
}
</style>
</head>
<body>

<div class="top-bar">
    <h1 onclick="showLibrary()">📝 Transcript Viewer</h1>
    <button class="nav-btn active" id="btnLibrary" onclick="showLibrary()">📚 Library</button>
    <button class="nav-btn" id="btnPlayer" onclick="showPlayer()">▶ Player</button>
    <div class="spacer"></div>
    <div class="info" id="pairInfo"></div>
</div>

<!-- ======== LIBRARY VIEW ======== -->
<div class="view library-view active" id="libraryView">
    <div class="library-toolbar">
        <input type="text" class="search-box" id="libSearch" placeholder="Search by filename or transcript content..." oninput="filterLibrary()">
        <select id="libFolderFilter" onchange="filterLibrary()">
            <option value="">All folders</option>
        </select>
        <span class="count" id="libCount"></span>
    </div>
    <div class="library-grid" id="libraryGrid">
        <div class="empty-state">
            <div class="icon">📚</div>
            <p>Loading library...</p>
        </div>
    </div>
</div>

<!-- ======== PLAYER VIEW ======== -->
<div class="view player-view" id="playerView">
    <div class="video-panel">
        <video id="videoPlayer" controls></video>
        <div class="video-controls">
            <button onclick="skip(-5)" title="Back 5s">◀ 5s</button>
            <button onclick="skip(5)" title="Forward 5s">5s ▶</button>
            <span class="time-display" id="timeDisplay">00:00 / 00:00</span>
            <div class="spacer"></div>
            <div class="speed-control">
                <span>Speed:</span>
                <select id="speedSelect" onchange="setSpeed(this.value)">
                    <option value="0.5">0.5×</option>
                    <option value="0.75">0.75×</option>
                    <option value="1" selected>1×</option>
                    <option value="1.25">1.25×</option>
                    <option value="1.5">1.5×</option>
                    <option value="2">2×</option>
                </select>
            </div>
        </div>
    </div>

    <div class="transcript-panel">
        <div class="transcript-header">
            <h2>Transcript</h2>
            <div class="spacer"></div>
            <input type="text" class="search-box" id="searchBox" placeholder="Search transcript..." oninput="filterTranscript()">
        </div>
        <div class="transcript-list" id="transcriptList">
            <div class="empty-state">
                <div class="icon">🎬</div>
                <p>Select a video from the Library to begin</p>
            </div>
        </div>
        <div class="kbd-hints">
            <span><kbd>Space</kbd> Play/Pause</span>
            <span><kbd>←</kbd> -5s</span>
            <span><kbd>→</kbd> +5s</span>
            <span><kbd>↑</kbd> Prev segment</span>
            <span><kbd>↓</kbd> Next segment</span>
            <span><kbd>Esc</kbd> Back to Library</span>
        </div>
    </div>
</div>

<script>
const video = document.getElementById('videoPlayer');
const transcriptList = document.getElementById('transcriptList');
const timeDisplay = document.getElementById('timeDisplay');
const searchBox = document.getElementById('searchBox');
const pairInfo = document.getElementById('pairInfo');
const libraryGrid = document.getElementById('libraryGrid');
const libSearch = document.getElementById('libSearch');
const libFolderFilter = document.getElementById('libFolderFilter');
const libCount = document.getElementById('libCount');

let segments = [];
let currentSegIdx = -1;
let libraryData = [];

// ---- View Switching ----
function showLibrary() {
    document.getElementById('libraryView').classList.add('active');
    document.getElementById('playerView').classList.remove('active');
    document.getElementById('btnLibrary').classList.add('active');
    document.getElementById('btnPlayer').classList.remove('active');
    video.pause();
    pairInfo.textContent = '';
}

function showPlayer() {
    document.getElementById('libraryView').classList.remove('active');
    document.getElementById('playerView').classList.add('active');
    document.getElementById('btnLibrary').classList.remove('active');
    document.getElementById('btnPlayer').classList.add('active');
}

// ---- Library ----
async function loadLibrary() {
    const res = await fetch('/api/library');
    libraryData = await res.json();

    // Populate folder filter
    const folders = [...new Set(libraryData.map(d => d.folder))].sort();
    libFolderFilter.innerHTML = '<option value="">All folders</option>';
    folders.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        libFolderFilter.appendChild(opt);
    });

    renderLibrary(libraryData);
}

function renderLibrary(items) {
    libraryGrid.innerHTML = '';
    if (!items.length) {
        libraryGrid.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><p>No matching files</p></div>';
        libCount.textContent = '0 files';
        return;
    }

    libCount.textContent = `${items.length} files`;

    items.forEach(item => {
        const row = document.createElement('div');
        row.className = 'lib-row';
        row.innerHTML = `
            <span class="lib-date">${escapeHtml(item.folder)}</span>
            <span class="lib-name">${escapeHtml(item.name)}</span>
            <span class="lib-summary">${escapeHtml(item.summary || '(no transcript)')}</span>
            <span class="lib-badge ${item.has_video ? 'has-video' : 'no-video'}">${item.has_video ? '🎥 video' : 'no video'}</span>
        `;
        row.addEventListener('click', () => openFromLibrary(item.folder, item.srt_name));
        libraryGrid.appendChild(row);
    });
}

function filterLibrary() {
    const query = libSearch.value.toLowerCase().trim();
    const folderFilter = libFolderFilter.value;

    const filtered = libraryData.filter(item => {
        if (folderFilter && item.folder !== folderFilter) return false;
        if (!query) return true;
        return (item.name.toLowerCase().includes(query) ||
                (item.summary || '').toLowerCase().includes(query));
    });

    renderLibrary(filtered);
}

async function openFromLibrary(folder, srtName) {
    showPlayer();
    pairInfo.textContent = `${folder} / ${srtName}`;
    await loadPair(folder, srtName);
}

// ---- Player: Load Pair ----
async function loadPair(folder, name) {
    const res = await fetch('/api/pair?folder=' + encodeURIComponent(folder) + '&name=' + encodeURIComponent(name));
    const data = await res.json();

    // Video
    if (data.video_url) {
        video.src = data.video_url;
        video.load();
    } else {
        video.removeAttribute('src');
    }

    // SRT
    if (data.srt_url) {
        const srtRes = await fetch(data.srt_url);
        const srtText = await srtRes.text();
        segments = parseSRT(srtText);
        renderTranscript();
    } else {
        segments = [];
        transcriptList.innerHTML = '<div class="empty-state"><div class="icon">📄</div><p>No SRT file found</p></div>';
    }
}

// ---- SRT Parser ----
function parseSRT(text) {
    const segs = [];
    const blocks = text.trim().split(/\n\s*\n/);
    for (const block of blocks) {
        const lines = block.trim().split('\n');
        if (lines.length < 3) continue;
        const tsMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
        if (!tsMatch) continue;
        const start = +tsMatch[1]*3600 + +tsMatch[2]*60 + +tsMatch[3] + +tsMatch[4]/1000;
        const end = +tsMatch[5]*3600 + +tsMatch[6]*60 + +tsMatch[7] + +tsMatch[8]/1000;
        const text = lines.slice(2).join(' ').trim();
        segs.push({ start, end, text });
    }
    return segs;
}

// ---- Render ----
function renderTranscript() {
    transcriptList.innerHTML = '';
    currentSegIdx = -1;
    segments.forEach((seg, i) => {
        const row = document.createElement('div');
        row.className = 'seg-row';
        row.dataset.idx = i;

        const timeFmt = formatTime(seg.start);
        row.innerHTML = `
            <span class="seg-time">${timeFmt}</span>
            <span class="seg-text">${escapeHtml(seg.text)}</span>
        `;
        row.addEventListener('click', () => seekTo(seg.start, i));
        transcriptList.appendChild(row);
    });
}

function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function formatTimeFull(s) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return `${h}:${m.toString().padStart(2,'0')}:${sec.toString().padStart(2,'0')}`;
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ---- Seek & Highlight ----
function seekTo(time, idx) {
    video.currentTime = time;
    highlightRow(idx);
}

function highlightRow(idx) {
    const rows = transcriptList.querySelectorAll('.seg-row');
    rows.forEach(r => r.classList.remove('active'));
    if (idx >= 0 && idx < rows.length) {
        rows[idx].classList.add('active');
        rows[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
        currentSegIdx = idx;
    }
}

// Auto-highlight on video timeupdate
video.addEventListener('timeupdate', () => {
    const t = video.currentTime;
    timeDisplay.textContent = `${formatTimeFull(t)} / ${formatTimeFull(video.duration || 0)}`;

    for (let i = segments.length - 1; i >= 0; i--) {
        if (t >= segments[i].start) {
            if (i !== currentSegIdx) highlightRow(i);
            break;
        }
    }
});

// ---- Controls ----
function skip(dt) { video.currentTime = Math.max(0, video.currentTime + dt); }
function setSpeed(v) { video.playbackRate = parseFloat(v); }

// ---- Search/Filter ----
function filterTranscript() {
    const query = searchBox.value.toLowerCase().trim();
    const rows = transcriptList.querySelectorAll('.seg-row');
    rows.forEach((row, i) => {
        const textEl = row.querySelector('.seg-text');
        const origText = segments[i].text;

        if (!query) {
            row.classList.remove('hidden');
            textEl.innerHTML = escapeHtml(origText);
            return;
        }

        const lower = origText.toLowerCase();
        if (lower.includes(query)) {
            row.classList.remove('hidden');
            const regex = new RegExp('(' + query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
            textEl.innerHTML = escapeHtml(origText).replace(regex, '<mark>$1</mark>');
        } else {
            row.classList.add('hidden');
        }
    });
}

// ---- Keyboard ----
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    // Escape goes back to library
    if (e.key === 'Escape') {
        e.preventDefault();
        showLibrary();
        return;
    }

    // Only handle player shortcuts when player is visible
    if (!document.getElementById('playerView').classList.contains('active')) return;

    switch(e.key) {
        case ' ':
            e.preventDefault();
            video.paused ? video.play() : video.pause();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            skip(-5);
            break;
        case 'ArrowRight':
            e.preventDefault();
            skip(5);
            break;
        case 'ArrowUp':
            e.preventDefault();
            if (currentSegIdx > 0) seekTo(segments[currentSegIdx - 1].start, currentSegIdx - 1);
            break;
        case 'ArrowDown':
            e.preventDefault();
            if (currentSegIdx < segments.length - 1) seekTo(segments[currentSegIdx + 1].start, currentSegIdx + 1);
            break;
    }
});

// Init
loadLibrary();
</script>
</body>
</html>"""


# ============================================================
# HTTP Server
# ============================================================

class TranscriptHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve the app, API endpoints, and media files."""

    video_root = ""
    transcript_root = ""

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self._serve_html()
        elif path == '/api/folders':
            self._api_folders()
        elif path == '/api/files':
            self._api_files(params)
        elif path == '/api/library':
            self._api_library()
        elif path == '/api/pair':
            self._api_pair(params)
        elif path.startswith('/media/video/'):
            self._serve_media(path, self.video_root, '/media/video/')
        elif path.startswith('/media/srt/'):
            self._serve_media(path, self.transcript_root, '/media/srt/')
        else:
            self.send_error(404)

    def _serve_html(self):
        content = HTML_PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def _api_folders(self):
        """List date folders available in the transcript root."""
        folders = []
        if os.path.isdir(self.transcript_root):
            for name in sorted(os.listdir(self.transcript_root)):
                if os.path.isdir(os.path.join(self.transcript_root, name)):
                    folders.append(name)
        self._send_json(folders)

    def _api_files(self, params):
        """List SRT files in a specific folder, with video availability."""
        folder = params.get('folder', [''])[0]
        srt_dir = os.path.join(self.transcript_root, folder)
        files = []

        if os.path.isdir(srt_dir):
            for name in sorted(os.listdir(srt_dir)):
                if os.path.splitext(name)[1].lower() in TRANSCRIPT_EXTENSIONS:
                    base = os.path.splitext(name)[0]
                    has_video = any(
                        os.path.exists(os.path.join(self.video_root, folder, base + ext))
                        for ext in VIDEO_EXTENSIONS
                    )
                    files.append({
                        "name": name,
                        "has_video": has_video,
                    })

        self._send_json(files)

    def _api_library(self):
        """List ALL transcript files across all folders with summaries from .txt."""
        library = []

        if not os.path.isdir(self.transcript_root):
            self._send_json(library)
            return

        for folder_name in sorted(os.listdir(self.transcript_root)):
            folder_path = os.path.join(self.transcript_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for srt_name in sorted(os.listdir(folder_path)):
                if os.path.splitext(srt_name)[1].lower() not in TRANSCRIPT_EXTENSIONS:
                    continue

                base = os.path.splitext(srt_name)[0]

                # Read summary from .txt file (first 150 chars)
                txt_path = os.path.join(folder_path, base + ".txt")
                summary = ""
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                            raw = f.read(300).strip()
                            # Clean up — collapse whitespace
                            summary = " ".join(raw.split())[:150]
                    except Exception:
                        summary = "(error reading)"

                # Check video
                has_video = any(
                    os.path.exists(os.path.join(self.video_root, folder_name, base + ext))
                    for ext in VIDEO_EXTENSIONS
                )

                library.append({
                    "folder": folder_name,
                    "name": base,
                    "srt_name": srt_name,
                    "summary": summary,
                    "has_video": has_video,
                })

        self._send_json(library)

    def _api_pair(self, params):
        """Get URLs for video + SRT pair."""
        folder = params.get('folder', [''])[0]
        name = params.get('name', [''])[0]
        base = os.path.splitext(name)[0]

        result = {"video_url": None, "srt_url": None}

        # Find video
        for ext in VIDEO_EXTENSIONS:
            vid_path = os.path.join(self.video_root, folder, base + ext)
            if os.path.exists(vid_path):
                result["video_url"] = f"/media/video/{folder}/{base}{ext}"
                break

        # SRT
        srt_path = os.path.join(self.transcript_root, folder, name)
        if os.path.exists(srt_path):
            result["srt_url"] = f"/media/srt/{folder}/{name}"

        self._send_json(result)

    def _serve_media(self, url_path, root, prefix):
        """Serve a media file with range request support for video scrubbing."""
        rel_path = unquote(url_path[len(prefix):])
        file_path = os.path.join(root, rel_path)

        if not os.path.isfile(file_path):
            self.send_error(404)
            return

        file_size = os.path.getsize(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

        try:
            # Handle Range requests (required for video seeking)
            range_header = self.headers.get('Range')
            if range_header:
                range_match = range_header.strip().replace('bytes=', '')
                parts = range_match.split('-')
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header('Content-Type', mime_type)
                self.send_header('Content-Length', length)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header('Content-Type', mime_type)
                self.send_header('Content-Length', file_size)
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # Browser cancelled the request — normal during seeking


def main():
    parser = argparse.ArgumentParser(
        description="Local transcript viewer with video sync.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT,
                        help=f"Root folder for video files (default: {DEFAULT_VIDEO_ROOT})")
    parser.add_argument("--transcript-root", default=DEFAULT_TRANSCRIPT_ROOT,
                        help=f"Root folder for SRT transcripts (default: {DEFAULT_TRANSCRIPT_ROOT})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to serve on (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    if not os.path.isdir(args.video_root):
        print(f"[WARNING] Video root not found: {args.video_root}")
    if not os.path.isdir(args.transcript_root):
        print(f"[ERROR] Transcript root not found: {args.transcript_root}")
        sys.exit(1)

    TranscriptHandler.video_root = os.path.abspath(args.video_root)
    TranscriptHandler.transcript_root = os.path.abspath(args.transcript_root)

    server = HTTPServer(('localhost', args.port), TranscriptHandler)
    print(f"\n{'='*50}")
    print(f"  Transcript Viewer")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*50}")
    print(f"  Videos:      {TranscriptHandler.video_root}")
    print(f"  Transcripts: {TranscriptHandler.transcript_root}")
    print(f"{'='*50}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
