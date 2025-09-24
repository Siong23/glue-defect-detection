import os
import time
import signal
import threading
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

from flask import Flask, render_template, request, redirect, url_for, send_file, Response, flash, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import cv2

# Try TurboJPEG for faster JPEG; fall back to cv2.imencode
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _jpeg = TurboJPEG()
    def encode_jpeg(img, quality=80):
        return _jpeg.encode(img, quality=quality, pixel_format=TJPF_BGR)
except Exception:
    _jpeg = None
    def encode_jpeg(img, quality=80):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""

# Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Torch
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try: torch.set_num_threads(1)
except Exception: pass
cv2.setNumThreads(1)

# GStreamer / Argus
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# ============================== Persistent CSI Camera ==============================
class PersistentCSICamera:
    def __init__(self, out_w=1280, out_h=720, fr_num=30, fr_den=1, stale_sec=2.0):
        self.out_w = out_w
        self.out_h = out_h
        self.fr_num = fr_num
        self.fr_den = fr_den
        self.stale_sec = stale_sec

        self._pipe = None
        self._sink = None
        self._bus = None

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_ts = 0.0

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _build_pipeline(self):
        pipe = Gst.Pipeline.new("csi-pipeline")

        src = Gst.ElementFactory.make("nvarguscamerasrc", "src")
        if src is None:
            raise RuntimeError("Failed to create nvarguscamerasrc")
        src.set_property("sensor-id", 0)
        try: src.set_property("bufapi-version", True)
        except Exception: pass

        # Mild ISP; safe defaults
        for k, v in [("tnr-mode", 1), ("tnr-strength", 0.5)]:
            try: src.set_property(k, v)
            except Exception: pass
        for k, v in [("ee-mode", 1), ("ee-strength", 0.3)]:
            try: src.set_property(k, v)
            except Exception: pass
        try: src.set_property("aeantibanding", 2)  # 60 Hz
        except Exception: pass

        caps_sensor = Gst.ElementFactory.make("capsfilter", "caps_sensor")
        caps_sensor.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM), width={self.out_w}, height={self.out_h}, "
                f"framerate={self.fr_num}/{self.fr_den}, format=NV12"
            ),
        )

        conv1 = Gst.ElementFactory.make("nvvidconv", "conv1")
        conv1.set_property("flip-method", 0)

        caps_bgrx = Gst.ElementFactory.make("capsfilter", "caps_bgrx")
        caps_bgrx.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, width={self.out_w}, height={self.out_h}, format=BGRx"
            ),
        )

        q1 = Gst.ElementFactory.make("queue", "q1")
        q1.set_property("leaky", 2)
        q1.set_property("max-size-buffers", 2)

        conv2 = Gst.ElementFactory.make("videoconvert", "conv2")

        caps_bgr = Gst.ElementFactory.make("capsfilter", "caps_bgr")
        caps_bgr.set_property("caps", Gst.Caps.from_string("video/x-raw, format=BGR"))

        sink = Gst.ElementFactory.make("appsink", "sink")
        sink.set_property("max-buffers", 2)
        sink.set_property("drop", True)
        sink.set_property("sync", False)

        for e in (src, caps_sensor, conv1, caps_bgrx, q1, conv2, caps_bgr, sink):
            pipe.add(e)

        assert src.link(caps_sensor)
        assert caps_sensor.link(conv1)
        assert conv1.link(caps_bgrx)
        assert caps_bgrx.link(q1)
        assert q1.link(conv2)
        assert conv2.link(caps_bgr)
        assert caps_bgr.link(sink)

        return pipe, sink

    def _start(self):
        self._pipe, self._sink = self._build_pipeline()
        self._bus = self._pipe.get_bus()
        ret = self._pipe.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipe.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer pipeline failed to start")
        self._bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE)
        self._last_frame_ts = time.time()

    def _really_null(self):
        try:
            if self._pipe:
                self._pipe.set_state(Gst.State.NULL)
                self._pipe.get_state(Gst.CLOCK_TIME_NONE)
        except Exception:
            pass
        finally:
            self._pipe = None
            self._sink = None
            self._bus = None

    def _run(self):
        timeout_ns = 500_000_000  # 0.5s
        backoff = 0.25
        fail_count = 0
        while not self._stop_evt.is_set():
            try:
                self._start()
                fail_count = 0
                while not self._stop_evt.is_set():
                    msg = self._bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                    if msg:
                        raise RuntimeError("GStreamer EOS/ERROR")

                    sample = self._sink.emit("try-pull-sample", timeout_ns)
                    if sample is None:
                        if time.time() - self._last_frame_ts > self.stale_sec:
                            raise RuntimeError("Stale frames watchdog")
                        continue

                    buf = sample.get_buffer()
                    caps = sample.get_caps()
                    w = caps.get_structure(0).get_value("width")
                    h = caps.get_structure(0).get_value("height")
                    ok, mapinfo = buf.map(Gst.MapFlags.READ)
                    if not ok:
                        continue
                    try:
                        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
                    finally:
                        buf.unmap(mapinfo)

                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._last_frame_ts = time.time()

            except Exception:
                self._really_null()
                fail_count += 1
                time.sleep(min(1.0, backoff * (1.5 ** min(fail_count, 8))))
                continue
            finally:
                self._really_null()

    def start(self):
        if not self._thread.is_alive():
            self._stop_evt.clear()
            self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        self._really_null()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

# ============================== App / Paths ==============================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
NEW_DIR = BASE_DIR / "new_images"
WEIGHTS_PATH = BASE_DIR / "weights" / "glue_best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
NEW_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-me"

# Start camera (30 fps preview; live detection uses same frames)
CAM = PersistentCSICamera(out_w=1280, out_h=720, fr_num=30, fr_den=1, stale_sec=2.0)
CAM.start()

# ============================== Model / Inference ==============================
CLASS_NAMES = ["Good", "NG"]
CONF_THR = 0.25
LIVE_CONF_THR = 0.85

_model = None
def get_model():
    global _model
    if _model is None:
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. `pip install ultralytics`")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")
        _model = YOLO(str(WEIGHTS_PATH))
        _model.to(DEVICE)
        if DEVICE == "cuda":
            try: _model.model.half()
            except Exception: pass
        try: torch.backends.cudnn.benchmark = True
        except Exception: pass
    return _model

def draw_detections(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], clss: List[int], confs: List[float]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    box_thick = max(2, int(min(w, h) / 200))
    font_scale = max(0.7, min(w, h) / 800.0)
    for (x1, y1, x2, y2), c, cf in zip(boxes, clss, confs):
        label = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"id{c}"
        color = (36, 255, 12) if label.lower().startswith("g") else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        txt = f"{label} {cf:.2f}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, box_thick))
        y_text = max(0, y1 - th - bl - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + bl + 6), color, -1)
        cv2.putText(out, txt, (x1 + 3, y_text + th),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(2, box_thick), cv2.LINE_AA)
    return out

def draw_status_badge(img: np.ndarray, label: str, conf: Optional[float]):
    h, w = img.shape[:2]
    base = max(1, min(w, h) / 640.0)
    font_scale = 0.9 * base
    thickness  = max(2, int(2 * base))
    pad        = int(12 * base)
    txt = "No Confident detection" if conf is None else f"{label}  {conf:.2f}"
    color = (36, 255, 12) if label.lower().startswith("g") else (0, 0, 255)
    (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    W, H = tw + 2*pad, th + bl + 2*pad
    x0, y0 = int(10 * base), int(10 * base)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + W, y0 + H), color, -1)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
    cv2.putText(img, txt, (x0 + pad, y0 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def run_inference_on_bgr(bgr_img: np.ndarray, conf_thr: float = CONF_THR):
    model = get_model()
    with torch.inference_mode():
        results = model.predict(bgr_img, conf=conf_thr, verbose=False, device=DEVICE, imgsz=640, iou=0.5)[0]
    boxes, clss, confs = [], [], []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        cls  = results.boxes.cls.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            clss.append(int(c))
            confs.append(float(cf))
    drawn = draw_detections(bgr_img, boxes, clss, confs)
    return drawn, boxes, clss, confs

# ============================== Live Detection Worker ==============================
class DetectionWorker:
    def __init__(self, cam: PersistentCSICamera, conf_thr=LIVE_CONF_THR, imgsz=512, rate_hz=10):
        self.cam = cam
        self.conf_thr = conf_thr
        self.imgsz = imgsz
        self.period = 1.0 / max(1, rate_hz)

        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

        self._lock = threading.Lock()
        self._overlay: Optional[np.ndarray] = None

    def start(self):
        if not self._thr.is_alive():
            self._stop.clear()
            self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=2.0)

    def get_overlay(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._overlay is None:
                return None
            return self._overlay.copy()

    def _run(self):
        model = get_model()
        hist = deque(maxlen=8)
        next_t = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(max(0, next_t - now))
            next_t = time.time() + self.period

            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            det_in = frame

            with torch.inference_mode():
                res = model.predict(det_in, conf=self.conf_thr, verbose=False,
                                    device=DEVICE, imgsz=self.imgsz, iou=0.5)[0]

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
            else:
                xyxy = np.empty((0,4), dtype=int)
                cls  = np.empty((0,), dtype=int)
                conf = np.empty((0,), dtype=float)

            out = frame.copy()
            ih, iw = out.shape[:2]
            box_thick = max(2, int(min(iw, ih) / 200))
            font_scale = max(0.7, min(iw, ih) / 800.0)

            best_good, best_ng = 0.0, 0.0
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                if cf < self.conf_thr:
                    continue
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
                color = (36, 255, 12) if name.lower().startswith("g") else (0, 0, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
                txt = f"{name} {cf:.2f}"
                (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, box_thick))
                y_text = max(0, y1 - th - bl - 6)
                cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + bl + 6), color, -1)
                cv2.putText(out, txt, (x1 + 3, y_text + th),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(2, box_thick), cv2.LINE_AA)

                if name.lower().startswith("g"):
                    best_good = max(best_good, float(cf))
                else:
                    best_ng = max(best_ng, float(cf))

            label_for_badge = "Good"; badge_conf: Optional[float] = None
            if best_ng >= self.conf_thr:
                label_for_badge = "NG";   badge_conf = best_ng
            elif best_good >= self.conf_thr:
                label_for_badge = "Good"; badge_conf = best_good

            hist.append((label_for_badge, badge_conf))
            labels = [h[0] for h in hist]
            label_for_badge = max(set(labels), key=labels.count)
            confs = [h[1] for h in hist if h[1] is not None and h[0] == label_for_badge]
            badge_conf = (sorted(confs)[len(confs)//2] if confs else None)

            draw_status_badge(out, label_for_badge, badge_conf)

            with self._lock:
                self._overlay = out

DETECTOR = DetectionWorker(CAM, conf_thr=LIVE_CONF_THR, imgsz=512, rate_hz=10)
DETECTOR.start()

# ============================== Routes ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live")
def live_page():
    return render_template("live.html")

# ---- Image Upload (unchanged) ----
UPLOAD_DIR.mkdir(exist_ok=True)
@app.route("/image", methods=["GET", "POST"])
def image_page():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part"); return redirect(request.url)
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file"); return redirect(request.url)
        ext_ok = f.filename.rsplit(".",1)[-1].lower() in {"png","jpg","jpeg","bmp"}
        if f and ext_ok:
            from_path = UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{secure_filename(f.filename)}"
            f.save(str(from_path))
            img = cv2.imdecode(np.fromfile(str(from_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                flash("Failed to read image"); return redirect(request.url)

            drawn, boxes, clss, confs = run_inference_on_bgr(img, conf_thr=CONF_THR)
            out_name = from_path.stem + "_result.jpg"
            (RESULTS_DIR / out_name).write_bytes(encode_jpeg(drawn, quality=85))
            return render_template(
                "image.html",
                result_image=url_for("static_result", filename=out_name),
                original_image=url_for("static_upload", filename=from_path.name),
                detections=[{"box": b, "cls": CLASS_NAMES[c] if 0<=c<len(CLASS_NAMES) else str(c), "conf": f"{cf:.2f}"} for b,c,cf in zip(boxes, clss, confs)]
            )
        else:
            flash("Unsupported file type (png/jpg/jpeg/bmp)."); return redirect(request.url)
    return render_template("image.html", result_image=None, original_image=None, detections=None)

# ---- Streams using CAM ----
def mjpeg_from_cam(target_fps=30, quality=80, source="raw"):
    next_due = 0.0
    while True:
        if source == "detect":
            frame = DETECTOR.get_overlay()
            if frame is None:
                frame = CAM.get_frame()
        else:
            frame = CAM.get_frame()

        if frame is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera starting…", (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            data = encode_jpeg(blank, quality=70)
        else:
            data = encode_jpeg(frame, quality=quality)

        now = time.time()
        if now < next_due:
            time.sleep(max(0, next_due - now))
        next_due = time.time() + 1.0 / max(1, target_fps)

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

@app.route("/video_feed_detect")
def video_feed_detect():
    return Response(mjpeg_from_cam(target_fps=30, quality=80, source="detect"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_capture")
def video_feed_capture():
    # Capture preview uses the SAME CAM frames (no second pipeline)
    return Response(mjpeg_from_cam(target_fps=30, quality=80, source="raw"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ---- New Capture & Save Page (no detection) ----
@app.route("/capture", methods=["GET"])
def capture_page():
    return render_template("capture.html")

@app.route("/save_capture", methods=["POST"])
def save_capture():
    # Grab a frame from the existing CAM
    t0 = time.time()
    frame = CAM.get_frame()
    # If CAM just started, wait a bit for first frame
    tries = 0
    while frame is None and tries < 50:
        time.sleep(0.02)
        frame = CAM.get_frame()
        tries += 1
    if frame is None:
        return jsonify({"status": "error", "message": "Camera not ready"})

    filename = f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    (NEW_DIR / filename).write_bytes(encode_jpeg(frame, quality=90))
    return jsonify({"status": "success", "filename": filename})

# ---- static helpers
@app.route("/uploads/<path:filename>")
def static_upload(filename):
    return send_file(UPLOAD_DIR / filename)

@app.route("/results/<path:filename>")
def static_result(filename):
    return send_file(RESULTS_DIR / filename)

# ----------- Graceful shutdown ----------
def _graceful_exit(*_):
    try: DETECTOR.stop()
    except Exception: pass
    try: CAM.stop()
    except Exception: pass
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    try:
        app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
    finally:
        _graceful_exit()

