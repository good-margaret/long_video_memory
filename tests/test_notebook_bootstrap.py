"""Portable setup tests; network and executable downloads are isolated from unit tests."""

import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memvid.demo import bootstrap as B


@pytest.fixture(autouse=True)
def clean_env():
    with patch.dict(os.environ):
        yield


def test_windows_ignores_anaconda_tools_and_reuses_managed_cache(tmp_path):
    def install(url, directory, names):
        assert "BtbN/FFmpeg-Builds" in url
        for name in names:
            (directory / name).write_bytes(b"tool")

    versions = {"ffmpeg": "ffmpeg version test", "ffprobe": "ffprobe version test"}
    with (
        patch.object(B.platform, "system", return_value="Windows"),
        patch.object(B.platform, "machine", return_value="AMD64"),
        patch.object(B, "_download_executables", side_effect=install) as download,
        patch.object(B, "_validate_tools", return_value=versions),
    ):
        first = B.setup_media_tools(tmp_path)
        second = B.setup_media_tools(tmp_path)
    assert first == second and download.call_count == 1
    assert Path(first["ffprobe"]).parent == tmp_path / "windows-x64"
    assert os.environ["MEMVID_FFPROBE"] == first["ffprobe"]


def test_linux_reuses_valid_system_tools(tmp_path):
    with (
        patch.object(B.platform, "system", return_value="Linux"),
        patch.object(B.platform, "machine", return_value="x86_64"),
        patch.object(B.shutil, "which", side_effect=lambda name: "/usr/bin/" + name),
        patch.object(
            B, "_validate_tools", return_value={"ffmpeg": "ffmpeg version test", "ffprobe": "ffprobe version test"}
        ),
        patch.object(B, "_download_executables") as download,
    ):
        tools = B.setup_media_tools(tmp_path)
        assert tools["ffmpeg"] == str(Path("/usr/bin/ffmpeg"))
        download.assert_not_called()


def test_invalid_system_tools_trigger_portable_fallback(tmp_path):
    def install(url, directory, names):
        for name in names:
            (directory / name).write_bytes(b"tool")

    with (
        patch.object(B.platform, "system", return_value="Linux"),
        patch.object(B.platform, "machine", return_value="x86_64"),
        patch.object(B.shutil, "which", side_effect=lambda name: "/bad/" + name),
        patch.object(
            B,
            "_validate_tools",
            side_effect=[RuntimeError("not native"), {"ffmpeg": "ffmpeg version x", "ffprobe": "ffprobe version x"}],
        ),
        patch.object(B, "_download_executables", side_effect=install),
    ):
        assert Path(B.setup_media_tools(tmp_path)["ffmpeg"]).parent == tmp_path / "linux-x64"


def test_zip_extraction_only_installs_named_tools(tmp_path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("release/bin/ffmpeg", b"ffmpeg")
        z.writestr("release/bin/ffprobe", b"ffprobe")
        z.writestr("../outside.txt", b"never extract this")
    target = tmp_path / "managed"
    B._download_executables(source.as_uri(), target, ["ffmpeg", "ffprobe"])
    assert (target / "ffprobe").read_bytes() == b"ffprobe"
    assert not (tmp_path / "outside.txt").exists()
    assert not (target / "download.partial.zip").exists()


def test_old_node_triggers_official_portable_deno_download(tmp_path):
    def install(url, directory, names):
        assert "github.com/denoland/deno/releases/" in url
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(b"deno")

    results = [SimpleNamespace(returncode=0, stdout="v20.1.0"), SimpleNamespace(returncode=0, stdout="deno 2.6.0")]
    with (
        patch.object(B.platform, "system", return_value="Linux"),
        patch.object(B.platform, "machine", return_value="x86_64"),
        patch.object(B.shutil, "which", side_effect=lambda name: "/bin/node" if name == "node" else None),
        patch.object(B.subprocess, "run", side_effect=results),
        patch.object(B, "_download_executables", side_effect=install),
    ):
        runtime = B.setup_js_runtime(tmp_path)
    assert Path(runtime["deno"]["path"]).is_file()


def test_download_includes_audio_and_checks_final_streams(tmp_path):
    yt = pytest.importorskip("yt_dlp")
    options = []

    class Downloader:
        def __init__(self, opts):
            options.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            assert download
            (tmp_path / "KbahC-QCKU8.h720.mp4").write_bytes(b"media")
            return {"id": "KbahC-QCKU8", "title": "test", "height": 720}

        def prepare_filename(self, info):
            return str(tmp_path / "KbahC-QCKU8.h720.mp4")

    with (
        patch.object(yt, "YoutubeDL", Downloader),
        patch.object(B, "setup_js_runtime", return_value={"node": {"path": "node"}}) as runtime,
        patch("memvid.video.probe.probe_video", return_value=SimpleNamespace(has_audio=True, duration_s=2448.6)),
    ):
        tools = {"ffmpeg": str(tmp_path / "bin/ffmpeg"), "ffprobe": str(tmp_path / "bin/ffprobe")}
        path = B.download_youtube("https://www.youtube.com/watch?v=KbahC-QCKU8", tmp_path, tools=tools)
        # A cached, verified download does not require JS setup or another network extraction.
        same = B.download_youtube("https://www.youtube.com/watch?v=KbahC-QCKU8", tmp_path, tools=tools)
    assert path == same and len(options) == 1 and runtime.call_count == 1
    assert "+ba" in options[0]["format"] and options[0]["noplaylist"]
    assert options[0]["ffmpeg_location"] == str(tmp_path / "bin")
    assert json.loads((tmp_path / "KbahC-QCKU8.h720.source.json").read_text())["actual_height"] == 720


def test_silent_youtube_result_is_rejected(tmp_path):
    yt = pytest.importorskip("yt_dlp")
    file = tmp_path / "silent.mp4"
    file.write_bytes(b"media")
    downloader = SimpleNamespace(extract_info=lambda *a, **k: {"id": "test"}, prepare_filename=lambda _: str(file))
    from contextlib import nullcontext

    with (
        patch.object(yt, "YoutubeDL", return_value=nullcontext(downloader)),
        patch.object(B, "setup_js_runtime", return_value={}),
        patch("memvid.video.probe.probe_video", return_value=SimpleNamespace(has_audio=False)),
        pytest.raises(RuntimeError, match="no audio"),
    ):
        B.download_youtube(
            "https://youtube.com/watch?v=test", tmp_path, tools={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
        )


def test_notebook_and_demo_messages_are_english():
    import re

    from memvid.demo import notebook

    assert not re.search("[А-Яа-яЁё]", Path(notebook.__file__).read_text(encoding="utf-8"))
    root = Path(__file__).parent
    if root.name == "tests":
        root = root.parent
    source = root / "notebooks/interactive_memory_demo.ipynb"
    if source.is_file():
        text = source.read_text(encoding="utf-8")
        assert not re.search("[А-Яа-яЁё]", text)
        assert "C:\\\\Users" not in text
        assert "S` does not mean speech" in text
