"""Portable notebook setup: managed media tools and optional YouTube download."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
import wave
import zipfile
from pathlib import Path


def _version(executable):
    p = subprocess.run([str(executable), "-version"], capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError(f"Cannot run {executable}: {p.stderr[-500:]}")
    return p.stdout.splitlines()[0]


def _validate_tools(ffmpeg, ffprobe, directory):
    versions = {"ffmpeg": _version(ffmpeg), "ffprobe": _version(ffprobe)}
    if not versions["ffmpeg"].lower().startswith("ffmpeg version") or not versions["ffprobe"].lower().startswith(
        "ffprobe version"
    ):
        raise RuntimeError("These executables are not native FFmpeg/FFprobe tools.")
    # Version output alone does not prove file access works (some Anaconda builds fail here).
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        sample = Path(tmp) / "check.wav"
        with wave.open(str(sample), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 1600)
        result = subprocess.run(
            [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(sample)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode or float(json.loads(result.stdout).get("format", {}).get("duration", 0)) <= 0:
            raise RuntimeError(f"FFprobe cannot read a local test file: {result.stderr[-500:]}")
    return versions


def _download_executables(url, directory, names):
    """Extract named binaries only; never extract archive paths or run installers."""
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "download.partial.zip"
    print(f"Downloading portable tools from {url}", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "memvid-notebook"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        with zipfile.ZipFile(archive) as z:
            for name in names:
                matches = [entry for entry in z.infolist() if Path(entry.filename).name == name and not entry.is_dir()]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected exactly one {name} in the tool archive")
                target = directory / name
                temporary = target.with_suffix(target.suffix + ".partial")
                with z.open(matches[0]) as source, temporary.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                temporary.chmod(0o755)
                temporary.replace(target)
    finally:
        archive.unlink(missing_ok=True)


def setup_media_tools(cache_dir, *, prefer_system=True):
    """Use working system tools on Unix; managed tools on Windows or as fallback.

    Sets MEMVID_FFMPEG/MEMVID_FFPROBE for this kernel and its worker processes.
    No user-specific paths or machine-wide PATH modifications are needed.
    """
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    arch = platform.machine().lower()
    arm = arch in {"arm64", "aarch64"}
    if system not in {"windows", "linux", "darwin"}:
        raise RuntimeError(f"Unsupported media-tool platform: {system}/{arch}")
    directory = root / f"{system}-{'arm64' if arm else 'x64'}"
    directory.mkdir(exist_ok=True)
    suffix = ".exe" if system == "windows" else ""
    ffmpeg, ffprobe = directory / f"ffmpeg{suffix}", directory / f"ffprobe{suffix}"
    if prefer_system and system != "windows":
        pair = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if all(pair):
            try:
                versions = _validate_tools(*pair, directory)
                ffmpeg, ffprobe = map(Path, pair)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                print(f"System media tools failed validation; using portable tools: {exc}")
    if ffmpeg.parent == directory:
        if not (ffmpeg.is_file() and ffprobe.is_file()):
            if system == "windows":
                if arm:
                    raise RuntimeError("Install native ARM64 FFmpeg/FFprobe or use an x64 environment.")
                url = (
                    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
                )
            else:
                tag = system + ("_arm64" if arm else "")
                url = f"https://github.com/zackees/ffmpeg_bins/raw/main/v8.0/{tag}.zip"
            _download_executables(url, directory, [ffmpeg.name, ffprobe.name])
        versions = _validate_tools(ffmpeg, ffprobe, directory)
    os.environ["MEMVID_FFMPEG"] = str(ffmpeg)
    os.environ["MEMVID_FFPROBE"] = str(ffprobe)
    result = {"ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe), "versions": versions}
    (root / "tools.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Media tools ready: {versions['ffmpeg']}; {versions['ffprobe']}")
    return result


def setup_js_runtime(cache_dir):
    """Discover supported Deno/Node, or download official portable Deno."""
    for name, minimum in [("deno", (2, 3)), ("node", (22, 0))]:
        path = shutil.which(name)
        if path:
            try:
                p = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                continue
            match = re.search(r"(\d+)\.(\d+)", p.stdout)
            if p.returncode == 0 and match and tuple(map(int, match.groups())) >= minimum:
                return {name: {"path": path}}
    system = platform.system().lower()
    arm = platform.machine().lower() in {"aarch64", "arm64"}
    architecture = "aarch64" if arm else "x86_64"
    target = {"windows": "pc-windows-msvc", "linux": "unknown-linux-gnu", "darwin": "apple-darwin"}.get(system)
    if not target:
        raise RuntimeError("Install Deno >=2.3 or Node >=22 to download from YouTube.")
    directory = Path(cache_dir).resolve() / f"deno-{architecture}-{target}"
    executable = directory / ("deno.exe" if system == "windows" else "deno")
    if not executable.exists():
        url = f"https://github.com/denoland/deno/releases/latest/download/deno-{architecture}-{target}.zip"
        _download_executables(url, directory, [executable.name])
    subprocess.run([str(executable), "--version"], check=True, capture_output=True, timeout=15)
    return {"deno": {"path": str(executable)}}


def download_youtube(url, raw_dir, *, tools, max_height=720, cache_dir=None):
    """Download one video with audio, validate both streams, and retain source metadata."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install the notebook extra, including yt-dlp[default].") from exc
    from memvid.video.probe import probe_video

    raw = Path(raw_dir).expanduser().resolve()
    raw.mkdir(parents=True, exist_ok=True)
    os.environ["MEMVID_FFMPEG"] = tools["ffmpeg"]
    os.environ["MEMVID_FFPROBE"] = tools["ffprobe"]
    if max_height is not None and int(max_height) < 1:
        raise ValueError("max_height must be positive or None")
    for record in raw.glob("*.source.json"):
        try:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict) or not isinstance(metadata.get("filename"), str):
            continue
        if metadata.get("url") == url and metadata.get("max_height") == max_height:
            cached = raw / Path(metadata["filename"]).name
            if cached.is_file() and probe_video(cached, compute_hash=False).has_audio:
                print(f"Using cached video with audio: {cached.name}")
                return cached
    runtimes = setup_js_runtime(cache_dir or raw / ".tools")
    limit = f"[height<={int(max_height)}]" if max_height else ""
    quality = f"h{int(max_height)}" if max_height else "best"
    options = {
        "format": f"bv*{limit}+ba/b{limit}",
        "merge_output_format": "mp4",
        "outtmpl": str(raw / f"%(id)s.{quality}.%(ext)s"),
        "noplaylist": True,
        "continuedl": True,
        "ffmpeg_location": str(Path(tools["ffmpeg"]).parent),
        "js_runtimes": runtimes,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            expected = Path(downloader.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            "YouTube download failed. Update yt-dlp[default] and check Internet access. "
            "If YouTube requires sign-in or blocks Kaggle, download on your own computer "
            "and upload the media as Kaggle input. Original error: " + str(exc)
        ) from exc
    video_id = info["id"]
    candidates = [expected] + [raw / f"{video_id}.{quality}{ext}" for ext in (".mp4", ".mkv", ".webm", ".mov")]
    output = next((p for p in candidates if p.is_file() and p.suffix in {".mp4", ".mkv", ".webm", ".mov"}), None)
    if output is None:
        raise RuntimeError("Downloader completed, but the final merged video was not found.")
    video = probe_video(output, compute_hash=False)
    if not video.has_audio:
        raise RuntimeError(f"Downloaded file has no audio: {output}. Remove the incomplete file and rerun.")
    (raw / f"{video_id}.{quality}.source.json").write_text(
        json.dumps(
            {
                "url": url,
                "video_id": video_id,
                "title": info.get("title"),
                "duration_s": video.duration_s,
                "max_height": max_height,
                "actual_height": info.get("height"),
                "filename": output.name,
                "yt_dlp_version": yt_dlp.version.__version__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Video and audio ready: {output.name} ({video.duration_s:.2f} s)")
    return output
