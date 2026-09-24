"""
CCTV-style camera DVR server

  Phone (WebRTC: video+audio) -> this server -> HLS on disk (audio+video merged)
                                             -> viewer page (live + rewind + playback)

Disk layout:
  recordings/<camera>/<YYYY-MM-DD>/<HH-MM-SS>/index.m3u8   (playlist)
                                             /seg_00001.ts  (4-second segments)

Run:  python server.py
"""
import asyncio
import json
import logging
import os
import re
import shutil
import socket
import ssl
import time
from datetime import date, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
from aiohttp import web
from av.video.frame import PictureType
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

# ----------------------------- settings ---------------------------------
BASE = Path(__file__).parent
REC_DIR = BASE / "recordings"
PORT = int(os.getenv("PORT", 8443))
SEG_SECONDS = int(os.getenv("SEG_SECONDS", 4))          # length of each HLS segment
SESSION_MINUTES = int(os.getenv("SESSION_MINUTES", 60)) # new playlist every N minutes
KEEP_DAYS = int(os.getenv("KEEP_DAYS", 7))              # keep today + previous 6 days
MIN_FREE_GB = float(os.getenv("MIN_FREE_GB", 5))        # delete oldest day if disk is below this
VIDEO_BPS = int(os.getenv("VIDEO_BPS", 1_500_000))
FPS = 15
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dvr")

recorders: dict = {}   # camera name -> CamRecorder
peers: dict = {}       # camera name -> RTCPeerConnection


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", s or "") or "phone1"


def slot_key(now: datetime):
    """Which recording 'slot' (playlist) a moment belongs to."""
    return (now.date(), (now.hour * 60 + now.minute) // SESSION_MINUTES)


# =========================== recording ====================================
class HLSSession:
    """One playlist on disk. Audio + video are muxed together into the same segments."""

    def __init__(self, cam, width, height, has_audio):
        now = datetime.now()
        self.key = slot_key(now)
        self.folder = REC_DIR / cam / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.origin = 0.0          # media clock value of this session's first video frame
        self.last_key = -1e9
        self.last_v_t = -1.0
        self.last_a_t = -1.0
        self.errors = 0

        self.container = av.open(
            str(self.folder / "index.m3u8"), "w", format="hls",
            options={
                "hls_time": str(SEG_SECONDS),
                "hls_playlist_type": "event",   # playlist only grows -> viewer can rewind
                "hls_flags": "independent_segments+program_date_time+temp_file",
                "hls_segment_filename": str(self.folder / "seg_%05d.ts"),
            },
        )
        self.width, self.height = width, height
        self.v = self.container.add_stream("libx264", rate=FPS)
        self.v.width, self.v.height = width, height
        self.v.pix_fmt = "yuv420p"
        self.v.bit_rate = VIDEO_BPS
        self.v.codec_context.time_base = Fraction(1, 90000)
        self.v.options = {"preset": "veryfast", "tune": "zerolatency", "g": "600"}

        self.a = None
        if has_audio:
            self.a = self.container.add_stream("aac", rate=48000)
            self.a.bit_rate = 64000

    def fit(self, frame):
        """Make the frame match the encoder (phones may change resolution mid-stream)."""
        if frame.format.name != "yuv420p" or frame.width != self.width or frame.height != self.height:
            frame = frame.reformat(width=self.width, height=self.height, format="yuv420p")
        return frame

    def write_video(self, frame, t):
        frame.pts = int(t * 90000)
        frame.time_base = Fraction(1, 90000)
        if t - self.last_key >= SEG_SECONDS:      # keyframe every segment
            frame.pict_type = PictureType.I
            self.last_key = t
        else:
            frame.pict_type = PictureType.NONE
        self.last_v_t = t
        for pkt in self.v.encode(frame):
            self.container.mux(pkt)

    def write_audio(self, frame, t):
        frame.pts = int(t * 48000)
        frame.time_base = Fraction(1, 48000)
        self.last_a_t = t
        for pkt in self.a.encode(frame):
            self.container.mux(pkt)

    def close(self):
        for stream in (self.v, self.a):
            if stream is None:
                continue
            try:
                for pkt in stream.encode(None):
                    self.container.mux(pkt)
            except Exception as e:
                log.warning("flush error: %s", e)
        try:
            self.container.close()      # writes the last segment + #EXT-X-ENDLIST
        except Exception as e:
            log.warning("close error: %s", e)
        log.info("closed %s", self.folder)


class CamRecorder:
    """Reads the incoming WebRTC tracks and feeds them to the current HLSSession."""

    def __init__(self, cam):
        self.cam = cam
        self.tracks = {}
        self.tasks = []
        self.session = None
        self.base = {}
        self.last_frame = time.monotonic()

    def add_track(self, track):
        self.tracks[track.kind] = track
        loop = self._video_loop if track.kind == "video" else self._audio_loop
        self.tasks.append(asyncio.create_task(loop(track)))

    def _media_time(self, kind, frame):
        """Convert a frame's RTP timestamp to seconds on a shared local clock."""
        if kind not in self.base:
            self.base[kind] = (time.monotonic(), frame.pts, frame.time_base)
        wall0, pts0, tb = self.base[kind]
        return wall0 + float((frame.pts - pts0) * tb)

    def on_video_frame(self, frame):
        """HOOK for later: run motion detection / AI on frame.to_ndarray(format='bgr24')."""
        pass

    def _rotate(self, frame, mt):
        self.close_session()
        w, h = frame.width & ~1, frame.height & ~1
        self.session = HLSSession(self.cam, w, h, "audio" in self.tracks)
        self.session.origin = mt
        log.info("[%s] new recording session -> %s", self.cam, self.session.folder)

    def close_session(self):
        if self.session:
            self.session.close()
            self.session = None

    async def _video_loop(self, track):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            self.last_frame = time.monotonic()
            mt = self._media_time("video", frame)
            if self.session is None or self.session.key != slot_key(datetime.now()):
                self._rotate(frame, mt)
            s = self.session
            t = mt - s.origin
            if t <= s.last_v_t:
                continue
            try:
                self.on_video_frame(frame)
                s.write_video(s.fit(frame), t)
            except Exception as e:
                s.errors += 1
                if s.errors <= 3:
                    log.warning("[%s] video write error: %s", self.cam, e)

    async def _audio_loop(self, track):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            mt = self._media_time("audio", frame)
            s = self.session
            if s is None or s.a is None:
                continue
            t = mt - s.origin
            if t < 0 or t <= s.last_a_t:
                continue
            try:
                s.write_audio(frame, t)
            except Exception as e:
                s.errors += 1
                if s.errors <= 3:
                    log.warning("[%s] audio write error: %s", self.cam, e)

    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.close_session()


# =========================== signaling ====================================
async def offer(request):
    cam = safe_name(request.query.get("cam"))
    params = await request.json()

    # the same phone reconnecting: drop the old connection first
    old_pc, old_rec = peers.pop(cam, None), recorders.pop(cam, None)
    if old_rec:
        await old_rec.stop()
    if old_pc:
        await old_pc.close()

    pc = RTCPeerConnection()
    rec = CamRecorder(cam)
    peers[cam], recorders[cam] = pc, rec

    @pc.on("track")
    def on_track(track):
        log.info("[%s] %s track received", cam, track.kind)
        rec.add_track(track)

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("[%s] connection %s", cam, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await rec.stop()
            await pc.close()
            if recorders.get(cam) is rec:
                recorders.pop(cam, None)
                peers.pop(cam, None)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    await pc.setLocalDescription(await pc.createAnswer())
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


# =========================== API for the viewer ===========================
def is_live(cam, folder: Path) -> bool:
    rec = recorders.get(cam)
    return bool(rec and rec.session and rec.session.folder == folder)


async def api_cameras(request):
    cams = sorted(p.name for p in REC_DIR.iterdir() if p.is_dir())
    for c in recorders:
        if c not in cams:
            cams.append(c)
    return web.json_response({"cameras": cams, "online": list(recorders)})


async def api_days(request):
    cam = safe_name(request.query.get("cam"))
    root = REC_DIR / cam
    days = []
    if root.exists():
        for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
            sessions = []
            for s in sorted(p for p in d.iterdir() if p.is_dir()):
                if (s / "index.m3u8").exists():
                    sessions.append({
                        "name": s.name.replace("-", ":"),
                        "url": f"/recordings/{cam}/{d.name}/{s.name}/index.m3u8",
                        "live": is_live(cam, s),
                    })
            days.append({"date": d.name, "sessions": sessions})
    return web.json_response({"days": days})


async def api_live(request):
    cam = safe_name(request.query.get("cam"))
    rec = recorders.get(cam)
    if rec and rec.session and (rec.session.folder / "index.m3u8").exists():
        f = rec.session.folder
        return web.json_response({"url": f"/recordings/{cam}/{f.parent.name}/{f.name}/index.m3u8"})
    return web.json_response({"url": None})


# =========================== retention ====================================
def day_folders():
    out = []
    if REC_DIR.exists():
        for cam in REC_DIR.iterdir():
            if cam.is_dir():
                for d in cam.iterdir():
                    try:
                        out.append((date.fromisoformat(d.name), d))
                    except ValueError:
                        pass
    return out


def cleanup_old():
    """Delete day folders older than KEEP_DAYS, and oldest days if the disk is nearly full."""
    cutoff = date.today() - timedelta(days=KEEP_DAYS - 1)
    for d, path in day_folders():
        if d < cutoff:
            log.info("retention: deleting %s", path)
            shutil.rmtree(path, ignore_errors=True)
    while shutil.disk_usage(REC_DIR).free < MIN_FREE_GB * 1e9:
        old = sorted((x for x in day_folders() if x[0] < date.today()), key=lambda x: x[0])
        if not old:
            break
        log.info("disk low: deleting %s", old[0][1])
        shutil.rmtree(old[0][1], ignore_errors=True)


def repair_playlists():
    """After a crash/power cut, playlists may lack ENDLIST. Add it so they play as finished."""
    for _, day in day_folders():
        for pl in day.glob("*/index.m3u8"):
            text = pl.read_text(errors="ignore")
            if "#EXT-X-ENDLIST" not in text:
                with open(pl, "a") as f:
                    f.write("#EXT-X-ENDLIST\n")


async def retention_loop():
    while True:
        try:
            await asyncio.get_running_loop().run_in_executor(None, cleanup_old)
        except Exception as e:
            log.warning("retention error: %s", e)
        await asyncio.sleep(3600)


async def watchdog_loop():
    """If a phone stops sending (screen off, wifi lost), close its recording cleanly."""
    while True:
        await asyncio.sleep(5)
        for cam, rec in list(recorders.items()):
            if time.monotonic() - rec.last_frame > 20:
                log.info("[%s] no frames for 20s, closing", cam)
                await rec.stop()
                pc = peers.pop(cam, None)
                recorders.pop(cam, None)
                if pc:
                    await pc.close()


async def background(app):
    repair_playlists()
    tasks = [asyncio.create_task(retention_loop()), asyncio.create_task(watchdog_loop())]
    yield
    for t in tasks:
        t.cancel()
    for rec in list(recorders.values()):
        await rec.stop()
    await asyncio.gather(*(pc.close() for pc in list(peers.values())))


# =========================== web app ======================================
@web.middleware
async def no_cache_playlists(request, handler):
    resp = await handler(request)
    if request.path.endswith(".m3u8"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def make_app():
    REC_DIR.mkdir(exist_ok=True)
    app = web.Application(middlewares=[no_cache_playlists])
    app.cleanup_ctx.append(background)
    app.router.add_get("/", lambda r: web.FileResponse(BASE / "phone.html"))
    app.router.add_get("/viewer", lambda r: web.FileResponse(BASE / "viewer.html"))
    app.router.add_post("/offer", offer)
    app.router.add_get("/api/cameras", api_cameras)
    app.router.add_get("/api/days", api_days)
    app.router.add_get("/api/live", api_live)
    app.router.add_static("/recordings", REC_DIR, follow_symlinks=False)
    return app


if __name__ == "__main__":
    cert, key = BASE / "cert.pem", BASE / "key.pem"
    ctx, scheme = None, "http"
    if cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        scheme = "https"
    else:
        print("WARNING: cert.pem/key.pem not found -> running plain HTTP (phone camera will be blocked)")
    ip = lan_ip()
    print(f"\n  Phone  : {scheme}://{ip}:{PORT}/")
    print(f"  Viewer : {scheme}://{ip}:{PORT}/viewer\n")
    web.run_app(make_app(), host="0.0.0.0", port=PORT, ssl_context=ctx, print=None)