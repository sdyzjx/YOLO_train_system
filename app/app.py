from __future__ import annotations

import ast
import base64
import datetime as dt
import html
import json
import math
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gradio as gr

os.environ.setdefault("MPLBACKEND", "Agg")

from app.backend.artifacts import RunDescriptor
from app.backend.config import AppConfig
from app.backend.feature_maps import generate_feature_map_visualization, list_feature_map_records
from app.backend.monitoring import collect_gpu_metrics
from app.backend.models import TaskMode, TrainingTask
from app.backend.queue_manager import TrainingQueueManager
from app.backend.utils import parse_key_value_text
from app.backend.visualization import VisualizationPayload, iter_providers


config = AppConfig.from_env()
queue_manager = TrainingQueueManager(config=config)

QUEUE_HEADERS = ["任务ID", "模式", "状态", "模型", "创建时间", "备注"]
PERF_GPU_HEADERS = ["GPU", "利用率(%)", "显存使用(MB)", "显存占比(%)", "显存剩余(MB)", "温度(°C)", "功耗(W)"]
PERF_PROCESS_HEADERS = ["GPU", "PID", "用户", "进程", "线程", "显存占用(MB)", "系统内存(MB)", "CPU(%)", "命令行"]


def _load_background_image_uri() -> str:
    image_path = Path(__file__).with_name("background.png")
    try:
        data = image_path.read_bytes()
    except OSError:
        return ""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


BACKGROUND_IMAGE_URI = _load_background_image_uri()
BACKGROUND_IMAGE_CSS = (
    f'url("{BACKGROUND_IMAGE_URI}")' if BACKGROUND_IMAGE_URI else 'url("background.png")'
)

IMAGE_FILE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
TEXT_FILE_EXTS = {
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".cfg",
    ".ini",
    ".toml",
    ".sh",
    ".log",
}
MAX_EDITABLE_BYTES = 1_000_000
MAX_SCRIPT_SPECIAL_FIELDS = 10


def _default_feature_map_imgsz() -> int:
    try:
        raw_value = config.default_train_params.get("imgsz", 640)
    except AttributeError:
        return 640
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return 640

CUSTOM_CSS = """
@font-face {
    font-family: "MonacoWeb";
    src: url("https://fonts.cdnfonts.com/s/14575/Monaco.woff") format("woff");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}
:root {
    --font-hero: "MonacoWeb", "Monaco", "JetBrains Mono", "Fira Code", "PingFang SC", "Microsoft YaHei", "Heiti SC", "Noto Sans SC", sans-serif;
    --body-background-fill: #05050a;
    --body-text-color: #f0e6ec;
    --block-background-fill: #0b0a11;
    --block-border-color: #c97a9f33;
    --border-color-primary: #c97a9f;
    --color-accent: #c97a9f;
    --link-text-color: #f4a9c4;
    --button-primary-background-fill: #c97a9f;
    --button-primary-text-color: #050506;
}
body, .gradio-container,
.gradio-container * {
    font-family: var(--font-hero) !important;
}
body, .gradio-container {
    background-color: #05050a !important;
    color: #f0e6ec !important;
}
body::after {
    content: "";
    position: fixed;
    right: 2vh;
    bottom: 2vh;
    width: 35vmin;
    height: 35vmin;
    max-width: 35%;
    max-height: 35%;
    background-image: __BACKGROUND_IMAGE__;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: bottom right;
    opacity: 0.4;
    pointer-events: none;
    z-index: 3;
}
.gradio-container {
    position: relative;
    z-index: 1;
}
.gradio-container .gr-button.primary, .gradio-container button.primary {
    background: linear-gradient(120deg, #c97a9f, #d689ac) !important;
    color: #050506 !important;
    border: none !important;
}
.gradio-container .gr-button.secondary, .gradio-container button.secondary {
    border: 1px solid #a64565 !important;
    color: #f9dbe3 !important;
    background: #3b0d18 !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select, .gradio-container .gr-textbox textarea {
    background: #11101a !important;
    color: #f0e6ec !important;
    border: 1px solid #c97a9f55 !important;
}
.gradio-container .gradio-slider input {
    color: #f0e6ec !important;
}
.gradio-container input[type="checkbox"] {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #c97a9f88 !important;
    background-color: #11101a !important;
    accent-color: #c97a9f !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
.gradio-container input[type="checkbox"]:checked {
    background-color: #c97a9f !important;
    border-color: #ffd2e8 !important;
    box-shadow: 0 0 0 1px #ffd2e844;
}
.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: #2a2733 !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
    background: #f0e6ec !important;
    border: 1px solid #c97a9f !important;
}
.gradio-container input[type="range"]::-moz-range-track {
    background: #2a2733 !important;
}
.gradio-container input[type="range"]::-moz-range-thumb {
    background: #f0e6ec !important;
    border: 1px solid #c97a9f !important;
}
.gradio-container .gradio-checkbox-group label, .gradio-container .gradio-checkbox label {
    color: #f0e6ec !important;
}
.gradio-container .gradio-gallery,
.gradio-container .gr-gallery,
.gradio-container .gr-image {
    background: #08070f !important;
    border-color: #c97a9f33 !important;
}
.gradio-container button[role="tab"] {
    border: none !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #c97a9f !important;
    border-bottom: 2px solid #c97a9f !important;
}
.gradio-container button[role="tab"][aria-selected="false"] {
    color: #f0e6ec !important;
}
.gradio-container .gradio-tabs {
    border-bottom: 1px solid #1a1824 !important;
}
.zoom-hoverable {
    cursor: zoom-in;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    transform-origin: center center;
    will-change: transform;
}
.zoom-hoverable:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 30px rgba(12, 5, 15, 0.55);
}
.zoom-overlay {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 2vw;
    background: rgba(4, 3, 10, 0.92);
    backdrop-filter: blur(4px);
    z-index: 9999;
}
.zoom-overlay.visible {
    display: flex;
}
.zoom-overlay__image {
    max-width: 96vw;
    max-height: 96vh;
    cursor: grab;
    transition: transform 0.08s ease-out;
    transform-origin: center center;
    box-shadow: 0 30px 50px rgba(0, 0, 0, 0.65);
}
.zoom-overlay__image.dragging {
    cursor: grabbing;
}
body.zoom-overlay-open {
    overflow: hidden;
}
.feature-map-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.feature-map-list__item {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: stretch;
    background: rgba(15, 10, 24, 0.85);
    border: 1px solid rgba(217, 169, 207, 0.3);
    border-radius: 14px;
    padding: 14px 16px 16px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
}
.feature-map-list__thumb {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    overflow: hidden;
    background: #07060e;
}
.feature-map-list__thumb img {
    width: 100%;
    height: auto;
    max-height: 420px;
    object-fit: contain;
    border-radius: 10px;
    cursor: zoom-in;
}
.feature-map-list__name {
    font-weight: 600;
    color: #f8eaff;
    font-size: 0.95rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.feature-map-list__path {
    font-size: 0.78rem;
    color: rgba(248, 234, 255, 0.7);
    word-break: break-all;
}
.feature-map-container {
    max-height: 540px;
    overflow-y: auto;
    padding: 12px 6px 12px 2px;
}
.monitor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-top: 8px;
}
.monitor-panel {
    position: relative;
}
.monitor-data-input {
    display: none !important;
}
.monitor-card {
    background: linear-gradient(135deg, rgba(201, 122, 159, 0.22), rgba(8, 6, 14, 0.85));
    border: 1px solid rgba(201, 122, 159, 0.35);
    border-radius: 14px;
    padding: 14px 16px 16px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}
.monitor-card__label {
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
}
.monitor-card__value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #fdf3ff;
    margin-top: 6px;
}
.monitor-card__sub {
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.65);
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    gap: 8px;
}
.monitor-bar {
    height: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    overflow: hidden;
    margin-top: 8px;
}
.monitor-bar__fill {
    height: 100%;
    display: block;
    background: linear-gradient(120deg, #c97a9f, #f8c4dc);
}
.monitor-empty {
    padding: 24px;
    border-radius: 14px;
    border: 1px dashed rgba(255, 255, 255, 0.2);
    text-align: center;
    color: rgba(255, 255, 255, 0.75);
    margin-top: 12px;
}
""".replace("__BACKGROUND_IMAGE__", BACKGROUND_IMAGE_CSS)


ZOOM_JS = """
<script>
(function () {
  if (window.__globalZoomEnhancer__) {
    return;
  }
  window.__globalZoomEnhancer__ = true;

  const overlay = document.createElement("div");
  overlay.className = "zoom-overlay";
  const overlayImg = document.createElement("img");
  overlayImg.className = "zoom-overlay__image";
  overlay.appendChild(overlayImg);

  let overlayMounted = false;

  const mountOverlay = () => {
    if (overlayMounted || !document.body) {
      return;
    }
    document.body.appendChild(overlay);
    overlayMounted = true;
  };

  let scale = 1;
  let transX = 0;
  let transY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;

  const applyTransform = () => {
    overlayImg.style.transform = `translate3d(${transX}px, ${transY}px, 0) scale(${scale})`;
  };

  const closeOverlay = () => {
    overlay.classList.remove("visible");
    document.body.classList.remove("zoom-overlay-open");
  };

  const openOverlay = (img) => {
    mountOverlay();
    const src = img.dataset?.zoomSrc || img.currentSrc || img.src;
    if (!src) return;
    scale = 1;
    transX = 0;
    transY = 0;
    overlayImg.src = src;
    overlay.classList.add("visible");
    document.body.classList.add("zoom-overlay-open");
    applyTransform();
  };
  window.__triggerZoomOverlay = openOverlay;
  window.__resetZoomOverlay = closeOverlay;

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      closeOverlay();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && overlay.classList.contains("visible")) {
      closeOverlay();
    }
  });

  overlayImg.addEventListener("click", (event) => event.stopPropagation());

  overlayImg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 0.15 : -0.15;
      scale = Math.min(6, Math.max(0.4, scale + delta));
      applyTransform();
    },
    { passive: false },
  );

  const handleMove = (event) => {
    if (!isDragging) return;
    transX = event.clientX - startX;
    transY = event.clientY - startY;
    applyTransform();
  };

  const stopDragging = () => {
    if (!isDragging) return;
    isDragging = false;
    overlayImg.classList.remove("dragging");
    window.removeEventListener("mousemove", handleMove);
    window.removeEventListener("mouseup", stopDragging);
  };

  overlayImg.addEventListener("mousedown", (event) => {
    event.preventDefault();
    isDragging = true;
    overlayImg.classList.add("dragging");
    startX = event.clientX - transX;
    startY = event.clientY - transY;
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", stopDragging);
  });

  overlayImg.addEventListener("dblclick", () => {
    scale = 1;
    transX = 0;
    transY = 0;
    applyTransform();
  });

  function attach(img) {
    if (
      !img ||
      img.dataset.zoomBound ||
      img.closest(".zoom-overlay") ||
      img.offsetWidth === 0 ||
      img.offsetHeight === 0
    ) {
      return;
    }
    img.dataset.zoomBound = "1";
    img.classList.add("zoom-hoverable");
    img.addEventListener("click", (event) => {
      event.stopPropagation();
      openOverlay(img);
    });
  }

  function scan() {
    document.querySelectorAll(".gradio-container img").forEach(attach);
  }

  const start = () => {
    mountOverlay();
    scan();
    if (!window.__globalZoomObserver__) {
      const observer = new MutationObserver(scan);
      observer.observe(document.body, { childList: true, subtree: true });
      window.__globalZoomObserver__ = observer;
    }
  };

  if (document.readyState !== "loading") {
    start();
  } else {
    window.addEventListener("DOMContentLoaded", start, { once: true });
  }
  window.addEventListener("gradio:ready", start);
})();
</script>
"""

MONITOR_JS = """
<script>
(function () {
  if (window.__perfMonitorUpdater__) {
    return;
  }
  window.__perfMonitorUpdater__ = true;

  const debugLog = (...args) => {
    try {
      console.debug("[perf-monitor]", ...args);
    } catch (_) {
      /* ignore */
    }
  };

  const PANEL_CONFIGS = [
    {
      kind: "host",
      gridSelector: "#monitor-grid-host",
      emptySelector: "#monitor-empty-host",
      dataSelector: ".monitor-data-host",
      defaultEmpty: "暂未获取到系统内存/CPU 信息。",
    },
    {
      kind: "gpu",
      gridSelector: "#monitor-grid-gpu",
      emptySelector: "#monitor-empty-gpu",
      dataSelector: ".monitor-data-gpu",
      defaultEmpty: "未检测到可用 GPU 或当前 GPU 空闲。",
    },
  ];

  const clampPercent = (value) => {
    const num = Number(value);
    if (Number.isNaN(num)) {
      return 0;
    }
    return Math.max(0, Math.min(100, num));
  };

  const ensureCardElement = (panel, key) => {
    const grid = document.querySelector(panel.gridSelector);
    if (!grid) {
      debugLog("grid not found", panel.kind, panel.gridSelector);
      return null;
    }
    if (!panel.__cardMap) {
      panel.__cardMap = new Map();
    }
    let card = panel.__cardMap.get(key);
    if (!card) {
      card = document.createElement("div");
      card.className = "monitor-card";
      const label = document.createElement("div");
      label.className = "monitor-card__label";
      const value = document.createElement("div");
      value.className = "monitor-card__value";
      card.appendChild(label);
      card.appendChild(value);
      card.__monitorElements = {
        label,
        value,
        rowEls: [],
        barEls: [],
      };
      panel.__cardMap.set(key, card);
    }
    grid.appendChild(card);
    return card;
  };

  const syncRowElements = (card, rows) => {
    const safeRows = Array.isArray(rows) ? rows : [];
    const elements = card.__monitorElements;
    elements.rowEls = elements.rowEls || [];
    safeRows.forEach((row, idx) => {
      let rowEl = elements.rowEls[idx];
      if (!rowEl) {
        rowEl = document.createElement("div");
        rowEl.className = "monitor-card__sub";
        const left = document.createElement("span");
        const right = document.createElement("span");
        rowEl.appendChild(left);
        rowEl.appendChild(right);
        const anchor = elements.barEls && elements.barEls[0] ? elements.barEls[0] : null;
        if (anchor && anchor.parentNode === card) {
          card.insertBefore(rowEl, anchor);
        } else {
          card.appendChild(rowEl);
        }
        elements.rowEls[idx] = rowEl;
      }
      const leftSpan = rowEl.children[0];
      const rightSpan = rowEl.children[1];
      if (leftSpan) {
        leftSpan.textContent = row && row.left ? row.left : "";
      }
      if (rightSpan) {
        rightSpan.textContent = row && row.right ? row.right : "";
      }
    });
    while (elements.rowEls.length > safeRows.length) {
      const removed = elements.rowEls.pop();
      if (removed) {
        removed.remove();
      }
    }
  };

  const syncBarElements = (card, bars) => {
    const safeBars = Array.isArray(bars) ? bars : [];
    const elements = card.__monitorElements;
    elements.barEls = elements.barEls || [];
    safeBars.forEach((bar, idx) => {
      let barEl = elements.barEls[idx];
      if (!barEl) {
        barEl = document.createElement("div");
        barEl.className = "monitor-bar";
        const fill = document.createElement("span");
        fill.className = "monitor-bar__fill";
        barEl.appendChild(fill);
        card.appendChild(barEl);
        elements.barEls[idx] = barEl;
      }
      const fill = barEl.firstElementChild;
      if (fill) {
        const pct = clampPercent(typeof bar === "number" ? bar : bar && bar.percent);
        fill.style.width = pct.toFixed(1) + "%";
      }
    });
    while (elements.barEls.length > safeBars.length) {
      const removed = elements.barEls.pop();
      if (removed) {
        removed.remove();
      }
    }
  };

  const applyCardData = (card, data) => {
    if (!card || !card.__monitorElements) {
      return;
    }
    card.__monitorElements.label.textContent = data && data.label ? data.label : "";
    card.__monitorElements.value.textContent = data && data.value ? data.value : "";
    syncRowElements(card, data && data.subs ? data.subs : []);
    syncBarElements(card, data && data.bars ? data.bars : []);
  };

  const updatePanel = (panel, payload) => {
    const grid = document.querySelector(panel.gridSelector);
    const emptyNode = document.querySelector(panel.emptySelector);
    if (!grid) {
      debugLog("skip update, grid missing", panel.kind);
      return;
    }
    const cards = payload && Array.isArray(payload.cards) ? payload.cards : [];
    const activeKeys = new Set();
    cards.forEach((cardData, idx) => {
      const key = (cardData && cardData.key) || `${panel.kind}-${idx}`;
      const cardEl = ensureCardElement(panel, key);
      if (!cardEl) {
        return;
      }
      activeKeys.add(key);
      applyCardData(cardEl, cardData || {});
    });
    if (panel.__cardMap) {
      panel.__cardMap.forEach((node, key) => {
        if (!activeKeys.has(key)) {
          node.remove();
          panel.__cardMap.delete(key);
        }
      });
    }
    if (!cards.length) {
      grid.style.display = "none";
      if (emptyNode) {
        emptyNode.textContent = (payload && payload.empty_text) || panel.defaultEmpty || "";
        emptyNode.style.display = "";
      }
      debugLog("no cards to display", panel.kind, payload);
    } else {
      grid.style.display = "";
      if (emptyNode) {
        emptyNode.style.display = "none";
      }
      debugLog("rendered cards", panel.kind, cards.length);
    }
  };

  const parsePayloadText = (text) => {
    if (!text.trim()) {
      return null;
    }
    try {
      return JSON.parse(text);
    } catch (err) {
      console.warn("无法解析性能监控数据：", err);
      return null;
    }
  };

  const readRawPayload = (panel) => {
    const container = document.querySelector(panel.dataSelector);
    if (!container) {
      debugLog("data container missing", panel.kind, panel.dataSelector);
      return "";
    }
    const field = container.querySelector("textarea, input, pre");
    if (field && "value" in field) {
      return field.value || "";
    }
    return field ? field.textContent || "" : container.textContent || "";
  };

  const startPolling = (panel) => {
    const tick = () => {
      const raw = readRawPayload(panel);
      if (raw && raw !== panel.__lastRaw) {
        panel.__lastRaw = raw;
        const payload = parsePayloadText(raw);
        updatePanel(panel, payload);
      }
    };
    tick();
    panel.__pollTimer = window.setInterval(tick, 400);
    debugLog("start polling", panel.kind);
  };

  const initWhenReady = () => {
    const ready = PANEL_CONFIGS.every(
      (panel) =>
        document.querySelector(panel.gridSelector) &&
        document.querySelector(panel.emptySelector) &&
        document.querySelector(panel.dataSelector),
    );
    if (!ready) {
      debugLog("panel not ready, retry…");
      requestAnimationFrame(initWhenReady);
      return;
    }
    debugLog("panels ready, attach polling");
    PANEL_CONFIGS.forEach(startPolling);
  };

  initWhenReady();
})();
</script>
"""

_SCRIPT_META_CACHE: Dict[str, Dict[str, Dict[str, object]]] = {}


def _load_script_special_keys_from_source(script_path: Path) -> Dict[str, Dict[str, object]]:
    if not script_path.exists():
        return {}
    try:
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        targets = []
        value_node = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        if not targets or value_node is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "SCRIPT_SPECIAL_KEYS":
                try:
                    value = ast.literal_eval(value_node)
                except Exception:  # noqa: BLE001
                    return {}
                return value if isinstance(value, dict) else {}
    return {}


def get_script_special_keys(script_name: str) -> Dict[str, Dict[str, object]]:
    if not script_name:
        return {}
    if script_name in _SCRIPT_META_CACHE:
        return _SCRIPT_META_CACHE[script_name]
    module_name = Path(script_name).stem
    script_path = config.train_script_dir / script_name
    try:
        module = import_module(f"train_method.{module_name}")
        meta = getattr(module, "SCRIPT_SPECIAL_KEYS", {})
    except Exception:  # noqa: BLE001
        meta = _load_script_special_keys_from_source(script_path)
    if not isinstance(meta, dict):
        meta = {}
    _SCRIPT_META_CACHE[script_name] = meta
    return meta


def _format_timestamp(value: dt.datetime) -> str:
    local = value.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S")


def available_models() -> List[Tuple[str, str]]:
    files = config.models_cfg_files()
    return [(f.name, f.name) for f in files]


def available_scripts() -> List[Tuple[str, str]]:
    return [(f.name, f.name) for f in config.available_train_scripts()]


def available_datasets(include_blank: bool = False) -> List[Tuple[str, str]]:
    files = config.dataset_cfg_files()
    choices = [(f.name, str(f.resolve())) for f in files]
    if include_blank:
        return [("不指定", "")] + choices
    return choices


def _resolve_selected_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    raw = Path(path_str)
    if not raw.is_absolute():
        raw = (config.base_dir / raw).resolve()
    else:
        raw = raw.resolve()
    try:
        raw.relative_to(config.base_dir)
    except ValueError:
        return None
    if not raw.exists():
        return None
    return raw


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_FILE_EXTS


def _is_text_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_FILE_EXTS


def _clamp_percent(value: float) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
    return value


def _serialize_monitor_payload(payload: Dict[str, Any]) -> str:
    sanitized = _sanitize_json_value(payload)
    try:
        data = json.dumps(sanitized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        data = json.dumps({}, ensure_ascii=False)
    return data


def _render_monitor_panel(kind: str, empty_text: str) -> str:
    grid_id = f"monitor-grid-{kind}"
    empty_id = f"monitor-empty-{kind}"
    return f"""
    <div class="monitor-panel" data-monitor-panel="{html.escape(kind)}">
        <div id="{empty_id}" class="monitor-empty">{html.escape(empty_text)}</div>
        <div id="{grid_id}" class="monitor-grid" aria-live="polite" style="display:none;"></div>
    </div>
    """


def _build_host_card_payload(host_stats: Optional[Dict[str, float]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cards": [],
        "empty_text": "暂未获取到系统内存/CPU 信息。",
    }
    if not host_stats:
        return payload
    timestamp_str = str(host_stats.get("timestamp", ""))[:19]
    specs = [
        (
            "host-cpu",
            "CPU 占用",
            f"{host_stats.get('cpu_percent', 0.0):.1f}%",
            f"1m 负载 {host_stats.get('load_1', 0.0):.2f}",
            host_stats.get("cpu_percent", 0.0),
        ),
        (
            "host-memory",
            "内存使用",
            f"{host_stats.get('memory_percent', 0.0):.1f}%",
            "{:.1f}/{:.1f} GB".format(
                host_stats.get("memory_used_gb", 0.0),
                host_stats.get("memory_total_gb", 0.0),
            ),
            host_stats.get("memory_percent", 0.0),
        ),
        (
            "host-swap",
            "Swap 使用",
            f"{host_stats.get('swap_percent', 0.0):.1f}%",
            "{:.1f}/{:.1f} GB".format(
                host_stats.get("swap_used_gb", 0.0),
                host_stats.get("swap_total_gb", 0.0),
            ),
            host_stats.get("swap_percent", 0.0),
        ),
    ]
    cards = []
    for key, label, value, extra, percent in specs:
        cards.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "subs": [
                    {
                        "left": str(extra),
                        "right": timestamp_str,
                    }
                ],
                "bars": [
                    {
                        "percent": _clamp_percent(percent or 0.0),
                    }
                ],
            }
        )
    payload["cards"] = cards
    payload["empty_text"] = f"最近更新时间 {timestamp_str}" if timestamp_str else payload["empty_text"]
    return payload


def _build_gpu_card_payload(gpu_stats: Sequence[Dict[str, object]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cards": [],
        "empty_text": "未检测到可用 GPU 或当前 GPU 空闲。",
    }
    cards: List[Dict[str, Any]] = []
    for gpu in gpu_stats:
        index = gpu.get("index", 0)
        util = _clamp_percent(gpu.get("utilization_gpu", 0.0))
        mem_total = _safe_float(gpu.get("memory_total_mb")) or 0.0
        mem_used = _safe_float(gpu.get("memory_used_mb")) or 0.0
        mem_pct = _clamp_percent((mem_used / mem_total * 100.0) if mem_total else 0.0)
        temperature_value = _safe_float(gpu.get("temperature"))
        power_value = _safe_float(gpu.get("power_w"))
        cards.append(
            {
                "key": f"gpu-{index}",
                "label": f"GPU {index}",
                "value": f"{util:.1f}%",
                "subs": [
                    {
                        "left": str(gpu.get("name", "GPU")),
                        "right": f"显存 {mem_used:.0f}/{mem_total:.0f} MB",
                    },
                    {
                        "left": f"显存占用 {mem_pct:.1f}%",
                        "right": "{} · {}".format(
                            "-" if temperature_value is None else f"{temperature_value:.0f}°C",
                            "-" if power_value is None else f"{power_value:.0f}W",
                        ),
                    },
                ],
                "bars": [
                    {"percent": util},
                    {"percent": mem_pct},
                ],
            }
        )
    payload["cards"] = cards
    return payload


def _read_text_file(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_EDITABLE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cast_special_value(value: str, value_type: str) -> tuple[Optional[str], Optional[str]]:
    text = (value or "").strip()
    if not text:
        return None, None
    value_type = (value_type or "text").lower()
    try:
        if value_type == "int":
            return str(int(float(text))), None
        if value_type == "float":
            return f"{float(text)}", None
        if value_type == "bool":
            normalized = text.lower()
            if normalized in {"true", "1", "yes", "y"}:
                return "true", None
            if normalized in {"false", "0", "no", "n"}:
                return "false", None
            raise ValueError("请输入 true/false")
        return text, None
    except ValueError as exc:
        return None, str(exc)


def _build_special_script_args(
    special_meta: Sequence[dict],
    special_values: Sequence[str],
) -> tuple[Dict[str, str], Optional[str]]:
    args: Dict[str, str] = {}
    for entry in special_meta:
        key = entry.get("key")
        if not key:
            continue
        index = entry.get("index", 0)
        raw_value = special_values[index] if index < len(special_values) else ""
        required = bool(entry.get("required"))
        label = entry.get("label", key)
        if not raw_value:
            if required:
                return {}, f"请填写脚本参数：{label}"
            continue
        casted, error = _cast_special_value(raw_value, entry.get("type"))
        if error:
            return {}, f"{label} 输入有误：{error}"
        if casted is None:
            if required:
                return {}, f"请填写脚本参数：{label}"
            continue
        args[key] = casted
    return args, None


def _build_parameters(
    model_file: Optional[str],
    data_path: Optional[str],
    epochs: Optional[int],
    batch: Optional[int],
    lr0: Optional[float],
    imgsz: Optional[int],
) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if model_file:
        params["model"] = model_file
    if data_path:
        params["data"] = data_path
    if epochs:
        params["epochs"] = str(epochs)
    if batch:
        params["batch"] = str(batch)
    if lr0:
        params["lr0"] = str(lr0)
    if imgsz:
        params["imgsz"] = str(imgsz)
    return params


def submit_quick_task(
    model_file: str,
    dataset_path: str,
    epochs: float,
    batch: float,
    lr0: float,
    imgsz: float,
    extra_params_text: str,
    user_note: str,
) -> str:
    if not model_file:
        return "请先选择模型配置文件。"
    if not dataset_path:
        return "请先选择数据集配置文件。"
    parameters = _build_parameters(
        model_file,
        dataset_path,
        int(epochs) if epochs else None,
        int(batch) if batch else None,
        lr0,
        int(imgsz) if imgsz else None,
    )
    extra = parse_key_value_text(extra_params_text)

    task = TrainingTask(
        mode=TaskMode.QUICK,
        parameters=parameters,
        extra_args=extra,
        user_note=user_note,
    )
    queue_manager.submit(task)
    return f"任务已提交，ID: {task.id}"


def submit_script_task(
    script_name: str,
    model_file: str,
    dataset_path: str,
    epochs: float,
    batch: float,
    lr0: float,
    imgsz: float,
    extra_params_text: str,
    script_params_text: str,
    user_note: str,
    special_field_state: Sequence[dict],
    *special_values: str,
) -> str:
    if not script_name:
        return "请先选择训练脚本。"
    script_path = config.train_script_dir / script_name
    if not script_path.exists():
        return f"脚本不存在: {script_path}"

    if script_name == "distill_p3p4_train.py" and not dataset_path:
        return "蒸馏脚本需要选择数据集配置文件。"

    parameters = _build_parameters(
        model_file,
        dataset_path,
        int(epochs) if epochs else None,
        int(batch) if batch else None,
        lr0,
        int(imgsz) if imgsz else None,
    )
    extra = parse_key_value_text(extra_params_text)
    script_args = parse_key_value_text(script_params_text)

    special_meta = list(special_field_state or [])
    special_values = list(special_values[:MAX_SCRIPT_SPECIAL_FIELDS])
    if special_meta:
        special_args, error = _build_special_script_args(special_meta, special_values)
        if error:
            return error
        script_args.update(special_args)

    if script_name == "distill_p3p4_train.py" and model_file:
        lowered = str(model_file).lower()
        if lowered.endswith(".pt"):
            script_args.setdefault("student_model", model_file)

    task = TrainingTask(
        mode=TaskMode.SCRIPT,
        parameters=parameters,
        extra_args=extra,
        script_path=script_path,
        script_args=script_args,
        user_note=user_note,
    )
    queue_manager.submit(task)
    return f"脚本任务已提交，ID: {task.id}"


def refresh_queue(selected_id: Optional[str]):
    tasks = queue_manager.list_tasks()
    rows: List[List[str]] = []
    dropdown_choices: List[tuple[str, str]] = []
    for task in tasks:
        rows.append(
            [
                task.id,
                task.mode.label,
                task.status.label,
                task.parameters.get("model", "-"),
                _format_timestamp(task.created_at),
                task.user_note or "-",
            ]
        )
        display = f"{task.status.label} | {task.parameters.get('model', '')} | {task.id[:6]}"
        dropdown_choices.append((display, task.id))

    value = selected_id if any(choice[1] == selected_id for choice in dropdown_choices) else (
        dropdown_choices[0][1] if dropdown_choices else None
    )

    return gr.update(value=rows, headers=QUEUE_HEADERS), gr.update(choices=dropdown_choices, value=value)


def _format_logs(task: TrainingTask) -> str:
    if not task.logs:
        return "暂无日志。"
    lines = [
        f"[{entry.timestamp.astimezone().strftime('%H:%M:%S')}] [{entry.level}] {entry.message}"
        for entry in task.logs[-500:]
    ]
    return "\n".join(lines)


def task_details(task_id: Optional[str]):
    if not task_id:
        return "请选择任务以查看详情。", "", {}
    task = queue_manager.get_task(task_id)
    if not task:
        return "未找到对应任务。", "", {}

    status_text = f"状态：{task.status.label}"
    if task.run_name:
        status_text += f"\nRun：{task.run_name}"
    validation = task.validation.metrics if task.validation else {}
    return status_text, _format_logs(task), validation


def cancel_selected_task(task_id: Optional[str]) -> str:
    if not task_id:
        return "请选择任务后再尝试停止。"
    message = queue_manager.cancel_task(task_id)
    return message


def list_runs() -> List[str]:
    descriptors = queue_manager.artifact_manager.list_all_runs()
    return [descriptor.name for descriptor in descriptors]


def _find_run_descriptor(run_name: str) -> Optional[RunDescriptor]:
    for descriptor in queue_manager.artifact_manager.list_all_runs():
        if descriptor.name == run_name:
            return descriptor
    return None


def weights_for_run(run_name: str) -> List[Tuple[str, str]]:
    descriptor = _find_run_descriptor(run_name)
    if not descriptor:
        return []
    run_dir = descriptor.path
    weights_dir = run_dir / "weights"
    preferred_files = []
    for name in ("best.pt", "last.pt"):
        path = weights_dir / name
        if path.exists():
            preferred_files.append((name, path))
    choices: List[Tuple[str, str]] = []
    if preferred_files:
        for label, path in preferred_files:
            choices.append((f"{label} ({run_name})", str(path.resolve())))
    else:
        for path in sorted(run_dir.rglob("*.pt")):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir)
            choices.append((f"{rel.as_posix()} ({run_name})", str(path.resolve())))
            if len(choices) >= 20:
                break
    return choices


def _image_to_data_uri(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    mime = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_feature_map_list(images: Sequence[str]) -> str:
    if not images:
        return ""
    items: List[str] = []
    for raw in images:
        path = Path(raw)
        label = path.name or str(path)
        data_uri = _image_to_data_uri(path)
        if not data_uri:
            continue
        safe_src = html.escape(data_uri)
        safe_label = html.escape(label)
        safe_full = html.escape(str(path))
        items.append(
            f"""
            <li class=\"feature-map-list__item\">
                <div class=\"feature-map-list__name\">{safe_label}</div>
                <div class=\"feature-map-list__thumb\">
                    <img src=\"{safe_src}\" data-zoom-src=\"{safe_src}\" alt=\"{safe_label}\" loading=\"lazy\"
                        onclick=\"window.__triggerZoomOverlay && window.__triggerZoomOverlay(this); event.stopPropagation();\" />
                </div>
                <div class=\"feature-map-list__path\">{safe_full}</div>
            </li>
            """
        )
    return "<ul class=\"feature-map-list\">" + "".join(items) + "</ul>"


def _strip_weight_label(value: str) -> str:
    """Remove the trailing '(run-name)' suffix that dropdown labels use."""
    value = value.strip()
    if " (" not in value or not value.endswith(")"):
        return value
    return value.rsplit(" (", 1)[0].strip()


def _resolve_weight_selection(weight_spec: str, descriptor: Optional[RunDescriptor]) -> Optional[Path]:
    """Translate a dropdown selection (e.g. 'best.pt (run)') into an actual weight file."""
    raw = (weight_spec or "").strip()
    if not raw:
        return None

    entries = [raw]
    stripped = _strip_weight_label(raw)
    if stripped and stripped not in entries:
        entries.append(stripped)

    candidates: List[Path] = []
    seen: Set[str] = set()

    def _add_candidate(path: Path):
        key = str(path.expanduser().resolve(strict=False))
        if key not in seen:
            seen.add(key)
            candidates.append(path.expanduser())

    for entry in entries:
        base = Path(entry).expanduser()
        _add_candidate(base)
        if descriptor and not base.is_absolute():
            weights_dir = descriptor.path / "weights"
            _add_candidate(descriptor.path / base)
            parts = base.parts
            if not parts or parts[0] != "weights":
                _add_candidate(weights_dir / base)
            name = base.name
            if name not in ("", "."):
                _add_candidate(weights_dir / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _feature_map_component_updates(records: Sequence[dict], preferred_id: Optional[str] = None) -> tuple[gr.Update, gr.Update]:
    if not records:
        return gr.update(choices=[], value=None, interactive=False), gr.update(value="", visible=False)
    choices = [(record.get("label", record.get("id", "")), record.get("id")) for record in records]
    valid_choices = [(label, value) for label, value in choices if value]
    if not valid_choices:
        return gr.update(choices=[], value=None, interactive=False), gr.update(value="", visible=False)
    target_id = None
    if preferred_id and any(value == preferred_id for _, value in valid_choices):
        target_id = preferred_id
    else:
        target_id = valid_choices[0][1]
    images = next((record.get("images", []) for record in records if record.get("id") == target_id), [])
    html_value = _render_feature_map_list(images)
    return (
        gr.update(choices=valid_choices, value=target_id, interactive=True),
        gr.update(value=html_value, visible=bool(images)),
    )


def _feature_map_records_for_descriptor(
    descriptor: Optional[RunDescriptor],
    preferred_id: Optional[str] = None,
) -> tuple[List[dict], gr.Update, gr.Update]:
    if not descriptor:
        return [], gr.update(choices=[], value=None, interactive=False), gr.update(value="", visible=False)
    records = list_feature_map_records(descriptor.path)
    dropdown_update, gallery_update = _feature_map_component_updates(records, preferred_id=preferred_id)
    return records, dropdown_update, gallery_update


def handle_feature_map_run_change(run_name: str):
    weight_update = update_weight_choices(run_name)
    descriptor = _find_run_descriptor(run_name)
    records, dropdown_update, gallery_update = _feature_map_records_for_descriptor(descriptor)
    return weight_update, dropdown_update, gallery_update, records, gr.update(value="")


def update_feature_map_gallery(selected_record: Optional[str], records_state: Sequence[dict]):
    records = list(records_state or [])
    if not records or not selected_record:
        return gr.update(value="", visible=False)
    images = next((record.get("images", []) for record in records if record.get("id") == selected_record), [])
    return gr.update(value=_render_feature_map_list(images), visible=bool(images))


def generate_feature_map_action(
    run_name: str,
    weight_path: str,
    manual_image_path: str,
    uploaded_image_path: Optional[str],
    current_records: Sequence[dict],
):
    records_snapshot = list(current_records or [])
    fallback_dropdown, fallback_gallery = _feature_map_component_updates(records_snapshot)
    if not run_name:
        return "请先选择 Run。", fallback_dropdown, fallback_gallery, records_snapshot
    descriptor = _find_run_descriptor(run_name)
    if not descriptor:
        return "未找到对应的 Run。", fallback_dropdown, fallback_gallery, records_snapshot
    weight_str = (weight_path or "").strip()
    if not weight_str:
        return "请先选择 Run 对应的权重文件。", fallback_dropdown, fallback_gallery, records_snapshot
    weight = _resolve_weight_selection(weight_str, descriptor)
    if not weight:
        return f"权重文件不存在：{weight_str}", fallback_dropdown, fallback_gallery, records_snapshot
    candidate_path = (manual_image_path or "").strip() or (uploaded_image_path or "")
    if not candidate_path:
        return "请上传或指定需要可视化的图片。", fallback_dropdown, fallback_gallery, records_snapshot
    image = Path(candidate_path)
    if not image.exists():
        return f"图像文件不存在：{image}", fallback_dropdown, fallback_gallery, records_snapshot
    imgsz_val = _default_feature_map_imgsz()
    try:
        latest_id, records = generate_feature_map_visualization(
            run_path=descriptor.path,
            weight_path=weight,
            image_path=image,
            imgsz=imgsz_val,
        )
    except Exception as exc:  # noqa: BLE001
        return f"生成特征图失败：{exc}", fallback_dropdown, fallback_gallery, records_snapshot
    dropdown_update, gallery_update = _feature_map_component_updates(records, preferred_id=latest_id)
    save_dir = descriptor.path / "feature_maps" / latest_id
    message = f"特征图已生成，保存目录：`{save_dir}`"
    return message, dropdown_update, gallery_update, records


def refresh_run_dropdowns():
    runs = list_runs()
    run_choices = runs
    selectable_runs = [("不选择", ""), *[(name, name) for name in runs]]
    feature_map_run_value = run_choices[0] if run_choices else None
    feature_map_run_update = gr.update(choices=run_choices, value=feature_map_run_value)
    feature_map_weight_update = update_weight_choices(feature_map_run_value) if feature_map_run_value else gr.update(choices=[], value=None)
    descriptor = _find_run_descriptor(feature_map_run_value) if feature_map_run_value else None
    records, record_dropdown_update, gallery_update = _feature_map_records_for_descriptor(descriptor)
    return (
        gr.update(choices=run_choices),
        gr.update(choices=selectable_runs, value=""),
        gr.update(choices=[], value=None),
        feature_map_run_update,
        feature_map_weight_update,
        record_dropdown_update,
        gallery_update,
        records,
        gr.update(value=""),
    )


def build_results_plot(data: Sequence[Dict[str, float]], metrics_to_show: Optional[Sequence[str]] = None):
    if not data:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    metrics = list(data)
    if not metrics:
        return None

    x_key = "epoch" if "epoch" in metrics[0] else next(iter(metrics[0].keys()), None)
    if not x_key:
        return None

    xs = [row.get(x_key, idx) for idx, row in enumerate(metrics)]
    available_keys = [key for key in metrics[0].keys() if key != x_key]
    targets = [key for key in (metrics_to_show or available_keys) if key in available_keys]
    if not targets:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for key in targets:
        ys = [row.get(key) for row in metrics]
        if any(value is None for value in ys):
            continue
        ax.plot(xs, ys, label=key)
    ax.set_xlabel(x_key)
    ax.set_title("Training Metrics Trend")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # release figure from pyplot state to avoid accumulating handles
    return fig


def _format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{value:.1f}"
    if abs_val >= 100:
        return f"{value:.2f}"
    if abs_val >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _sorted_metric_headers(keys: Sequence[str]) -> List[str]:
    def priority(key: str) -> tuple[int, str]:
        if key == "epoch":
            return (0, key)
        if key.startswith("train/"):
            return (1, key)
        if key.startswith("val/"):
            return (2, key)
        if key.startswith("metrics/"):
            return (3, key)
        if key.startswith("lr"):
            return (4, key)
        return (5, key)

    return sorted(keys, key=priority)


def _format_run_table(data: Sequence[Dict[str, float]]) -> tuple[Optional[List[List[str]]], Optional[List[str]]]:
    if not data:
        return None, None
    keys: set[str] = set()
    for row in data:
        keys.update(row.keys())
    if not keys:
        return None, None
    headers = _sorted_metric_headers(keys)
    formatted_rows: List[List[str]] = []
    for row in data:
        formatted_rows.append([_format_metric_value(row.get(header)) for header in headers])
    return formatted_rows, headers


def _compute_training_progress(descriptor: Optional[RunDescriptor], data: Sequence[Dict[str, float]]) -> float:
    if not descriptor or not data:
        return 0.0
    epochs_target: Optional[int] = None
    args = queue_manager.artifact_manager.load_args(descriptor.path)
    epochs_raw = (args or {}).get("epochs") if args else None
    if isinstance(epochs_raw, (int, float)):
        epochs_target = int(epochs_raw)
    elif isinstance(epochs_raw, str):
        try:
            epochs_target = int(float(epochs_raw))
        except ValueError:
            epochs_target = None

    last_epoch = max((row.get("epoch") for row in data if "epoch" in row), default=None)
    if last_epoch is None:
        return 0.0
    current_epoch = last_epoch + 1  # epochs usually 0-indexed
    if epochs_target and epochs_target > 0:
        progress = min(current_epoch / epochs_target, 1.0)
    else:
        progress = 1.0
    return round(progress * 100, 2)


def load_run_metrics(run_name: str, selected_metrics: Optional[List[str]] = None):
    if not run_name:
        return (
            gr.update(value=None, headers=None),
            None,
            {},
            gr.update(choices=[], value=[]),
            [],
            gr.update(value=0),
        )
    descriptor = _find_run_descriptor(run_name)
    if not descriptor:
        return (
            gr.update(value=None, headers=None),
            None,
            {},
            gr.update(choices=[], value=[]),
            [],
            gr.update(value=0),
        )
    data = queue_manager.artifact_manager.read_results(descriptor.path)
    formatted_rows, headers = _format_run_table(data)
    metric_keys = [key for key in (headers or []) if key != "epoch"]
    metrics_selection = [m for m in (selected_metrics or []) if m in metric_keys]
    if not metrics_selection and metric_keys:
        metrics_selection = metric_keys[: min(len(metric_keys), 4)] or metric_keys

    plot = build_results_plot(data, metrics_selection)
    payload = VisualizationPayload(run_name=descriptor.name, run_path=descriptor.path, metrics=data)
    extras = {provider.name: provider.render(payload) for provider in iter_providers()}
    progress = _compute_training_progress(descriptor, data)
    return (
        gr.update(value=formatted_rows, headers=headers),
        plot,
        extras,
        gr.update(choices=metric_keys, value=metrics_selection, interactive=True),
        data,
        gr.update(value=progress),
    )


def refresh_run_metrics(run_name: str, selected_metrics: Optional[List[str]], _current_data: Optional[Sequence[Dict[str, float]]] = None):
    if not run_name:
        return gr.update(value=None, headers=None), None, {}, [], gr.update(value=0)
    descriptor = _find_run_descriptor(run_name)
    if not descriptor:
        return gr.update(value=None, headers=None), None, {}, [], gr.update(value=0)
    data = queue_manager.artifact_manager.read_results(descriptor.path)
    formatted_rows, headers = _format_run_table(data)
    metrics_selection = selected_metrics or []
    plot = build_results_plot(data, metrics_selection)
    payload = VisualizationPayload(run_name=descriptor.name, run_path=descriptor.path, metrics=data)
    extras = {provider.name: provider.render(payload) for provider in iter_providers()}
    progress = _compute_training_progress(descriptor, data)
    return gr.update(value=formatted_rows, headers=headers), plot, extras, data, gr.update(value=progress)


def update_metric_plot(selected_metrics: List[str], cached_data: Sequence[Dict[str, float]], run_name: str):
    data = cached_data
    if not data and run_name:
        descriptor = _find_run_descriptor(run_name)
        if descriptor:
            data = queue_manager.artifact_manager.read_results(descriptor.path)
    return build_results_plot(data, selected_metrics)


def update_weight_choices(run_name: str):
    options = weights_for_run(run_name) if run_name else []
    return gr.update(choices=options, value=options[0][1] if options else None)


def handle_file_selection(selection: Optional[str | List[str]]):
    if isinstance(selection, list):
        raw_value = selection[0] if selection else ""
    else:
        raw_value = selection or ""
    raw_value = (raw_value or "").strip()
    resolved = _resolve_selected_path(raw_value) if raw_value else None
    display_path = str(resolved) if resolved else raw_value
    state_value = str(resolved) if resolved else ""

    info_lines: List[str] = []
    image_update = gr.update(value=None, visible=False)
    editor_update = gr.update(value="", visible=False)
    save_btn_update = gr.update(visible=False, interactive=False)
    download_btn_update = gr.update(visible=False, value=None, label="下载文件")
    if not raw_value:
        info_lines.append("请选择文件或目录。")
    elif not resolved:
        info_lines.append("文件不存在或不在项目根目录内。")
    else:
        stats = resolved.stat()
        modified = dt.datetime.fromtimestamp(stats.st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        size_kb = f"{stats.st_size / 1024:.1f} KB"
        kind = "目录" if resolved.is_dir() else "文件"
        info_lines.append(f"{kind} · {size_kb} · 更新于 {modified}")
        if resolved.is_dir():
            info_lines.append("无法在此预览或编辑目录。")
        else:
            download_btn_update = gr.update(
                value=str(resolved),
                visible=True,
                label=f"下载文件：{resolved.name}",
            )
            if _is_image_file(resolved):
                image_update = gr.update(value=str(resolved), visible=True)
            if _is_text_file(resolved):
                content = _read_text_file(resolved)
                if content is None:
                    info_lines.append("文件过大，暂不支持在线编辑。")
                else:
                    editor_update = gr.update(value=content, visible=True)
                    save_btn_update = gr.update(visible=True, interactive=True)
            else:
                info_lines.append("此文件类型不支持在线编辑。")

    info_text = "\n\n".join(info_lines) if info_lines else "请选择文件或目录。"
    return (
        display_path,
        state_value,
        info_text,
        image_update,
        editor_update,
        save_btn_update,
        download_btn_update,
        gr.update(value=""),
    )


def save_file_edits(path_str: str, content: str) -> str:
    path = _resolve_selected_path(path_str)
    if not path:
        return "未选择有效文件或文件不在项目目录内。"
    if path.is_dir():
        return "无法保存目录，请选择具体文件。"
    if not _is_text_file(path):
        return "当前文件类型不支持在线编辑。"
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"保存失败：{exc}"
    return f"已保存 {path.name} 于 {dt.datetime.now().strftime('%H:%M:%S')}"


def refresh_performance_monitor():
    metrics = collect_gpu_metrics()
    gpus = metrics.get("gpus", [])
    host_payload = _serialize_monitor_payload(_build_host_card_payload(metrics.get("host")))
    gpu_payload = _serialize_monitor_payload(_build_gpu_card_payload(gpus))
    provider = metrics.get("provider", "")
    provider_label = {
        "nvml": "NVML",
        "nvidia-smi": "nvidia-smi",
        "unavailable": "Unavailable",
    }.get(provider, str(provider).upper())
    status_lines = [
        f"**采样时间：** {metrics.get('timestamp', '')}",
        f"**数据源：** {provider_label}",
    ]
    message = metrics.get("message")
    if message:
        status_lines.append(message)
    if not gpus:
        status_lines.append("⚠️ 未检测到 GPU 或当前无法读取 GPU 指标。")

    gpu_rows: List[List[object]] = []
    for gpu in gpus:
        mem_total = _safe_float(gpu.get("memory_total_mb")) or 0.0
        mem_used = _safe_float(gpu.get("memory_used_mb")) or 0.0
        mem_free = _safe_float(gpu.get("memory_free_mb")) or max(mem_total - mem_used, 0.0)
        mem_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0
        gpu_rows.append(
            [
                f"GPU {gpu.get('index')} · {gpu.get('name', '')}",
                round(float(gpu.get("utilization_gpu") or 0.0), 2),
                round(mem_used, 2),
                round(mem_pct, 2),
                round(mem_free, 2),
                gpu.get("temperature") if gpu.get("temperature") not in (None, "") else "-",
                gpu.get("power_w") if gpu.get("power_w") not in (None, "") else "-",
            ]
        )

    process_rows: List[List[object]] = []
    for proc in metrics.get("processes", []):
        gpu_label = "-" if proc.get("gpu_index") in (None, "") else f"GPU {proc.get('gpu_index')}"
        process_rows.append(
            [
                gpu_label,
                proc.get("pid"),
                proc.get("user", "-"),
                proc.get("name", "-"),
                proc.get("threads", "-"),
                round(float(proc.get("gpu_memory_mb") or 0.0), 2),
                round(float(proc.get("ram_mb") or 0.0), 2),
                round(float(proc.get("cpu_percent") or 0.0), 2),
                proc.get("cmdline", "")[:160],
            ]
        )

    return "\n\n".join(status_lines), host_payload, gpu_payload, gpu_rows, process_rows


def handle_script_selection(script_name: str):
    metadata = get_script_special_keys(script_name)
    items = list(metadata.items())[:MAX_SCRIPT_SPECIAL_FIELDS]
    info_lines: List[str] = []
    component_updates: List[gr.Update] = []
    state_payload: List[dict] = []
    for idx in range(MAX_SCRIPT_SPECIAL_FIELDS):
        if idx < len(items):
            key, info = items[idx]
            label = info.get("label", key)
            default = info.get("default", "")
            required = bool(info.get("required"))
            param_type = info.get("type", "text")
            info_lines.append(f"- **{label}** (`{key}`) · 类型：{param_type}{' · 必填' if required else ''}")
            component_updates.append(
                gr.update(
                    visible=True,
                    label=f"{label} ({key})",
                    value=str(default) if default not in (None, "") else "",
                    placeholder=str(info.get("placeholder", "")),
                )
            )
            state_payload.append(
                {
                    "key": key,
                    "type": param_type,
                    "required": required,
                    "label": label,
                    "index": idx,
                }
            )
        else:
            component_updates.append(gr.update(visible=False, value="", label=f"脚本参数 {idx + 1}"))
    info_text = (
        "该脚本未提供额外参数，可使用“脚本特有参数 (key=value)”文本框手动输入。"
        if not info_lines
        else "可配置参数：\n" + "\n".join(info_lines)
    )
    return (
        gr.update(value=info_text),
        gr.update(visible=not info_lines),
        state_payload,
        *component_updates,
    )


def manual_validate_front(
    run_name: str,
    weight_option: str,
    custom_weight_path: str,
    dataset_path: str,
    imgsz: float,
    batch: float,
    extra_params_text: str,
    note: str,
) -> Tuple[str, Dict[str, float], str]:
    custom_path = (custom_weight_path or "").strip()
    weight_choice = (weight_option or "").strip()
    weight_path_str = custom_path or weight_choice
    if not weight_path_str:
        return "请提供权重文件路径或选择已有 Run。", {}, ""
    if not dataset_path:
        return "请先选择数据集配置文件。", {}, ""

    descriptor = _find_run_descriptor(run_name) if run_name else None
    weight_path = _resolve_weight_selection(weight_path_str, descriptor)
    if not weight_path:
        return f"权重文件不存在：{weight_path_str}", {}, ""
    dataset = Path(dataset_path).expanduser().resolve()
    extra = parse_key_value_text(extra_params_text)

    imgsz_val = int(imgsz) if imgsz else None
    batch_val = int(batch) if batch else None

    try:
        summary, logs, descriptor = queue_manager.manual_validate(
            weight_path=weight_path,
            dataset_path=dataset,
            imgsz=imgsz_val,
            batch=batch_val,
            note=note,
            extra_params=extra,
        )
    except FileNotFoundError as exc:
        return str(exc), {}, ""
    except Exception as exc:  # noqa: BLE001
        return f"验证过程中出错: {exc}", {}, ""

    status_label = "验证成功 ✅" if summary.success else "验证失败 ❌"
    lines = [
        status_label,
        f"保存目录：`{descriptor.path}`",
    ]
    if summary.metrics:
        metric_lines = [f"- **{key}**: {value:.4f}" for key, value in summary.metrics.items()]
        lines.append("### 指标")
        lines.extend(metric_lines)
    recent_logs = "\n".join(
        f"[{log.timestamp.astimezone().strftime('%H:%M:%S')}] [{log.level}] {log.message}"
        for log in logs[-20:]
    )
    raw_block = summary.raw_output.strip() or recent_logs
    if raw_block:
        lines.append("```\n" + raw_block + "\n```")
    log_text = "\n".join(
        f"[{entry.timestamp.astimezone().strftime('%H:%M:%S')}] [{entry.level}] {entry.message}" for entry in logs
    )
    return "\n".join(lines), summary.metrics, log_text


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="YOLO 训练管理系统",
        css=CUSTOM_CSS,
        head=f"{MONITOR_JS}\n{ZOOM_JS}",
    ) as demo:
        gr.Markdown("## YOLO 训练管理系统")

        with gr.Tab("任务队列"):
            queue_table = gr.Dataframe(
                value=[],
                headers=QUEUE_HEADERS,
                col_count=(len(QUEUE_HEADERS), "dynamic"),
                interactive=False,
                label="队列概览",
            )
            refresh_btn = gr.Button("刷新队列")
            task_selector = gr.Dropdown(label="选择任务", choices=[])
            cancel_btn = gr.Button("停止/取消任务", variant="secondary")
            cancel_result = gr.Markdown("")
            status_info = gr.Markdown()
            log_box = gr.Textbox(label="实时日志", lines=16, interactive=False)
            validation_json = gr.JSON(label="验证结果", value={})

        with gr.Tab("性能监控"):
            monitor_status_md = gr.Markdown("正在采集 GPU 信息…")
            host_cards_html = gr.HTML(
                value=_render_monitor_panel("host", "正在等待系统资源数据…"),
                label="系统资源概览",
            )
            host_cards_data = gr.Textbox(
                value=_serialize_monitor_payload({"cards": [], "empty_text": "正在等待系统资源数据…"}),
                show_label=False,
                interactive=False,
                lines=1,
                elem_classes=["monitor-data-input", "monitor-data-host"],
            )
            gpu_cards_html = gr.HTML(
                value=_render_monitor_panel("gpu", "正在等待 GPU 数据…"),
                label="GPU 状态",
            )
            gpu_cards_data = gr.Textbox(
                value=_serialize_monitor_payload({"cards": [], "empty_text": "正在等待 GPU 数据…"}),
                show_label=False,
                interactive=False,
                lines=1,
                elem_classes=["monitor-data-input", "monitor-data-gpu"],
            )
            gpu_table = gr.Dataframe(
                value=[],
                headers=PERF_GPU_HEADERS,
                col_count=(len(PERF_GPU_HEADERS), "dynamic"),
                interactive=False,
                label="GPU 即时状态",
            )
            process_table = gr.Dataframe(
                value=[],
                headers=PERF_PROCESS_HEADERS,
                col_count=(len(PERF_PROCESS_HEADERS), "dynamic"),
                interactive=False,
                label="占用 GPU 的进程",
            )
            monitor_timer = gr.Timer(value=2.0)

        with gr.Tab("快速添加"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=available_models(),
                    allow_custom_value=True,
                    label="模型配置 (models_cfg)",
                )
                dataset_dropdown = gr.Dropdown(
                    choices=available_datasets(),
                    label="数据集配置 (datasets/cfg)",
                )
            with gr.Row():
                epochs_slider = gr.Slider(1, 500, value=int(config.default_train_params["epochs"]), step=1, label="epochs")
                batch_slider = gr.Slider(1, 256, value=int(config.default_train_params["batch"]), step=1, label="batch")
                lr_slider = gr.Number(value=float(config.default_train_params["lr0"]), label="lr0")
                imgsz_slider = gr.Number(value=int(config.default_train_params["imgsz"]), label="imgsz")
            extra_box = gr.Textbox(
                label="额外参数 (一行一个，格式 key=value)",
                lines=4,
                placeholder="optimizer=SGD\npatience=20",
            )
            user_note_box = gr.Textbox(label="备注/自定义信息", placeholder="自定义 run 名称附加信息")
            submit_quick_btn = gr.Button("提交快速训练任务", variant="primary")
            quick_result = gr.Markdown("")

        with gr.Tab("脚本添加"):
            script_dropdown = gr.Dropdown(choices=available_scripts(), label="训练脚本 (train_method)")
            with gr.Row():
                script_model_dropdown = gr.Dropdown(
                    choices=[("不指定", "")] + available_models(),
                    allow_custom_value=True,
                    label="模型配置 (可选)",
                )
                script_dataset_dropdown = gr.Dropdown(
                    choices=available_datasets(include_blank=True),
                    label="数据集配置 (可选)",
                )
            with gr.Row():
                script_epochs_slider = gr.Slider(1, 500, value=int(config.default_train_params["epochs"]), step=1, label="epochs")
                script_batch_slider = gr.Slider(1, 256, value=int(config.default_train_params["batch"]), step=1, label="batch")
                script_lr_slider = gr.Number(value=float(config.default_train_params["lr0"]), label="lr0")
                script_imgsz_slider = gr.Number(value=int(config.default_train_params["imgsz"]), label="imgsz")
            extra_param_box = gr.Textbox(
                label="通用参数 (key=value)",
                lines=4,
                placeholder="device=0\nworkers=4",
            )
            script_param_box = gr.Textbox(
                label="脚本特有参数 (key=value)",
                lines=4,
                placeholder="-- 若脚本需要额外参数，请在此输入",
            )
            script_special_info_md = gr.Markdown(
                "该脚本未提供额外参数，可使用“脚本特有参数 (key=value)”文本框手动输入。"
            )
            script_special_inputs = [
                gr.Textbox(label=f"脚本参数 {idx + 1}", visible=False)
                for idx in range(MAX_SCRIPT_SPECIAL_FIELDS)
            ]
            script_special_state = gr.State([])
            script_note_box = gr.Textbox(label="备注/自定义信息")
            submit_script_btn = gr.Button("提交脚本训练任务", variant="primary")
            script_result = gr.Markdown("")

        with gr.Tab("运行监控"):
            run_selector = gr.Dropdown(label="选择 Run", choices=list_runs())
            run_refresh_btn = gr.Button("刷新 Run 列表")
            metric_selector = gr.CheckboxGroup(label="选择展示指标", choices=[], value=[])
            run_plot = gr.Plot(label="结果可视化")
            overall_progress = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=0.1,
                interactive=False,
                label="总体进度 (%)",
            )
            run_table = gr.Dataframe(value=None, interactive=False, label="结果数据")
            extra_json = gr.JSON(label="扩展可视化插件输出", value={})
            run_data_state = gr.State([])

        with gr.Tab("模型验证"):
            _available_runs = list_runs()
            val_run_dropdown = gr.Dropdown(
                label="选择训练 Run (可选)",
                choices=[("不选择", "")] + [(name, name) for name in _available_runs],
                value="",
            )
            val_weight_dropdown = gr.Dropdown(label="Run 权重文件", choices=[], allow_custom_value=True)
            custom_weight_input = gr.Textbox(label="自定义权重路径（优先级更高）", placeholder="/path/to/weights.pt")
            dataset_val_dropdown = gr.Dropdown(
                choices=available_datasets(),
                label="数据集配置 (datasets/cfg)",
            )
            with gr.Row():
                val_imgsz = gr.Number(value=float(config.default_train_params["imgsz"]), label="imgsz")
                val_batch = gr.Number(value=float(config.default_train_params["batch"]), label="batch")
            val_extra_params = gr.Textbox(label="额外参数 (key=value)", lines=4, placeholder="conf=0.25\niou=0.6")
            val_note_box = gr.Textbox(label="备注/自定义信息", placeholder="manual-val")
            val_button = gr.Button("开始验证", variant="primary")
            val_result_md = gr.Markdown("")
            val_metrics_json = gr.JSON(label="验证指标", value={})
            val_logs_box = gr.Textbox(label="验证日志", lines=12, interactive=False)
            with gr.Accordion("特征图可视化", open=False):
                feature_map_run_dropdown = gr.Dropdown(
                    label="Run",
                    choices=_available_runs,
                    value=_available_runs[0] if _available_runs else None,
                    allow_custom_value=False,
                )
                feature_map_weight_dropdown = gr.Dropdown(label="Run 权重文件", choices=[], allow_custom_value=True)
                feature_map_image_path = gr.Textbox(label="图片路径 (可选)", placeholder="/path/to/image.jpg")
                feature_map_image_input = gr.Image(
                    label="上传图片 (可选)",
                    type="filepath",
                    height=240,
                )
                feature_map_generate_btn = gr.Button("生成特征图", variant="primary")
                feature_map_status_md = gr.Markdown("")
                feature_map_record_dropdown = gr.Dropdown(label="历史可视化记录", choices=[], interactive=False)
                feature_map_gallery = gr.HTML(
                    value="",
                    visible=False,
                    label="特征图输出",
                    elem_classes=["feature-map-container"],
                )
                feature_map_records_state = gr.State([])

        file_path_state = gr.State("")
        with gr.Tab("文件浏览"):
            with gr.Row():
                file_explorer = gr.FileExplorer(
                    root_dir=str(config.base_dir),
                    label="项目文件",
                    height=400,
                    file_count="single",
                )
                with gr.Column():
                    selected_file_text = gr.Textbox(label="已选择路径", interactive=False)
                    copy_path_btn = gr.Button("复制路径", variant="secondary")
                    download_file_btn = gr.DownloadButton("下载文件", variant="secondary", visible=False)
                    file_info_md = gr.Markdown("请选择文件或目录。")
            with gr.Row():
                file_image_preview = gr.Image(
                    label="图片预览",
                    value=None,
                    interactive=False,
                    visible=False,
                    height=320,
                    elem_classes=["zoomable-media"],
                )
                file_editor = gr.Textbox(label="文件内容", value="", lines=20, interactive=True, visible=False)
            file_save_btn = gr.Button("保存文件修改", variant="primary", visible=False)
            file_action_status = gr.Markdown("")

        refresh_btn.click(
            refresh_queue,
            inputs=[task_selector],
            outputs=[queue_table, task_selector],
        )
        demo.load(
            refresh_queue,
            inputs=[task_selector],
            outputs=[queue_table, task_selector],
        )
        demo.load(
            refresh_performance_monitor,
            outputs=[monitor_status_md, host_cards_data, gpu_cards_data, gpu_table, process_table],
        )
        monitor_timer.tick(
            refresh_performance_monitor,
            outputs=[monitor_status_md, host_cards_data, gpu_cards_data, gpu_table, process_table],
        )
        demo.load(
            handle_feature_map_run_change,
            inputs=[feature_map_run_dropdown],
            outputs=[
                feature_map_weight_dropdown,
                feature_map_record_dropdown,
                feature_map_gallery,
                feature_map_records_state,
                feature_map_status_md,
            ],
        )
        task_selector.change(
            task_details,
            inputs=[task_selector],
            outputs=[status_info, log_box, validation_json],
        )
        cancel_btn.click(
            cancel_selected_task,
            inputs=[task_selector],
            outputs=[cancel_result],
        )

        submit_quick_btn.click(
            submit_quick_task,
            inputs=[
                model_dropdown,
                dataset_dropdown,
                epochs_slider,
                batch_slider,
                lr_slider,
                imgsz_slider,
                extra_box,
                user_note_box,
            ],
            outputs=[quick_result],
        )

        submit_script_btn.click(
            submit_script_task,
            inputs=[
                script_dropdown,
                script_model_dropdown,
                script_dataset_dropdown,
                script_epochs_slider,
                script_batch_slider,
                script_lr_slider,
                script_imgsz_slider,
                extra_param_box,
                script_param_box,
                script_note_box,
                script_special_state,
                *script_special_inputs,
            ],
            outputs=[script_result],
        )

        run_selector.change(
            load_run_metrics,
            inputs=[run_selector, metric_selector],
            outputs=[run_table, run_plot, extra_json, metric_selector, run_data_state, overall_progress],
        )

        run_refresh_btn.click(
            refresh_run_dropdowns,
            outputs=[
                run_selector,
                val_run_dropdown,
                val_weight_dropdown,
                feature_map_run_dropdown,
                feature_map_weight_dropdown,
                feature_map_record_dropdown,
                feature_map_gallery,
                feature_map_records_state,
                feature_map_status_md,
            ],
        )

        metric_selector.change(
            update_metric_plot,
            inputs=[metric_selector, run_data_state, run_selector],
            outputs=[run_plot],
        )
        feature_map_run_dropdown.change(
            handle_feature_map_run_change,
            inputs=[feature_map_run_dropdown],
            outputs=[
                feature_map_weight_dropdown,
                feature_map_record_dropdown,
                feature_map_gallery,
                feature_map_records_state,
                feature_map_status_md,
            ],
        )
        feature_map_record_dropdown.change(
            update_feature_map_gallery,
            inputs=[feature_map_record_dropdown, feature_map_records_state],
            outputs=[feature_map_gallery],
        )
        feature_map_generate_btn.click(
            generate_feature_map_action,
            inputs=[
                feature_map_run_dropdown,
                feature_map_weight_dropdown,
                feature_map_image_path,
                feature_map_image_input,
                feature_map_records_state,
            ],
            outputs=[
                feature_map_status_md,
                feature_map_record_dropdown,
                feature_map_gallery,
                feature_map_records_state,
            ],
        )

        script_dropdown.change(
            handle_script_selection,
            inputs=[script_dropdown],
            outputs=[
                script_special_info_md,
                script_param_box,
                script_special_state,
                *script_special_inputs,
            ],
        )

        val_run_dropdown.change(
            update_weight_choices,
            inputs=[val_run_dropdown],
            outputs=[val_weight_dropdown],
        )

        val_button.click(
            manual_validate_front,
            inputs=[
                val_run_dropdown,
                val_weight_dropdown,
                custom_weight_input,
                dataset_val_dropdown,
                val_imgsz,
                val_batch,
                val_extra_params,
                val_note_box,
            ],
            outputs=[val_result_md, val_metrics_json, val_logs_box],
        )

        file_explorer.change(
            handle_file_selection,
            inputs=[file_explorer],
            outputs=[
                selected_file_text,
                file_path_state,
                file_info_md,
                file_image_preview,
                file_editor,
                file_save_btn,
                download_file_btn,
                file_action_status,
            ],
        )

        copy_path_btn.click(
            fn=None,
            inputs=[file_path_state],
            outputs=None,
            js="(path) => navigator.clipboard && navigator.clipboard.writeText(path || '')",
        )

        file_save_btn.click(
            save_file_edits,
            inputs=[file_path_state, file_editor],
            outputs=[file_action_status],
        )

        timer = gr.Timer(config.realtime_refresh_sec)
        timer.tick(
            refresh_queue,
            inputs=[task_selector],
            outputs=[queue_table, task_selector],
        )
        timer.tick(
            task_details,
            inputs=[task_selector],
            outputs=[status_info, log_box, validation_json],
        )

        timer.tick(
            refresh_run_metrics,
            inputs=[run_selector, metric_selector, run_data_state],
            outputs=[run_table, run_plot, extra_json, run_data_state, overall_progress],
        )
    return demo


demo = build_interface()


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False)
