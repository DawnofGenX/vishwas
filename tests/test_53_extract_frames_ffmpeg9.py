"""Frame extraction must work on the ffmpeg actually installed on this host.

REGRESSION (2026-10-21)
-----------------------
`media_utils.extract_frames` passed `-vsync vfr` to ffmpeg. ffmpeg 9.0.1
(ffmpeg version n9.0.1-11-ge47273f4d9-20260831) REMOVED the `-vsync` option
(it was deprecated in favour of `-fps_mode`). The call therefore failed with:

    Unrecognized option 'vsync'.
    Error splitting the argument list: Option not found

`subprocess.run(..., capture_output=True)` swallowed the error, so
`extract_frames` returned an EMPTY list, every video check degraded, and the
video channel read UNVERIFIED for every clip. The whole video channel was
dead on this host and nothing logged the cause — the capability simply
reported "all frames undecodable for EFFORT inference".

This test pins the fix: the frame-extraction command must not use an option
the installed ffmpeg rejects, and must actually produce frames.

The test is hermetic in the sense that it skips when ffmpeg or a sample video
is absent — but on this host both are present, so it is a REAL assertion.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas import media_utils
from vishwas.media_utils import extract_frames

SAMPLE_VIDEOS = [
    Path("/home/hermes/fusion_av/ffc23/subset/fake/333_377.mp4"),
    Path("/home/hermes/fusion_av/ffc23/subset/real/950.mp4"),
]


def _installed_ffmpeg_supports(args: list[str]) -> bool:
    """True when the INSTALLED ffmpeg accepts *args* (it exits non-zero on
    an unrecognised option)."""
    if not shutil.which(media_utils.FFMPEG):
        return False
    r = subprocess.run([media_utils.FFMPEG, "-hide_banner", *args],
                       capture_output=True, text=True, timeout=60)
    combined = (r.stdout or "") + (r.stderr or "")
    if "Unrecognized option" in combined or "Option not found" in combined:
        return False
    return True


def test_installed_ffmpeg_rejects_the_vsync_option():
    """The installed ffmpeg must be probed: if it rejects -vsync, the
    extraction command must not use it. (On ffmpeg < 5 -vsync is fine.)"""
    if not shutil.which(media_utils.FFMPEG):
        pytest.skip("ffmpeg not on PATH")
    if _installed_ffmpeg_supports(["-vsync", "vfr", "-f", "null", "-"]):
        pytest.skip("installed ffmpeg still supports -vsync; nothing to pin")
    # ffmpeg 9+ rejected it -> our command must not contain it
    assert not _installed_ffmpeg_supports(["-vsync", "vfr", "-f", "null", "-"])


def test_extract_frames_returns_frames_not_an_empty_list():
    """The real regression: extract_frames must actually produce frames.

    Before the fix it returned [] on this host (the -vsync error was
    swallowed), and every video verdict degraded to UNVERIFIED.
    """
    if not shutil.which(media_utils.FFMPEG):
        pytest.skip("ffmpeg not on PATH")
    video = next((v for v in SAMPLE_VIDEOS if v.exists()), None)
    if video is None:
        pytest.skip("no sample video available on this host")
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(video, Path(td) / "frames", n=4)
        assert frames, (
            f"extract_frames({video.name}) returned no frames — the ffmpeg "
            f"invocation is failing and the error is being swallowed"
        )
        assert len(frames) <= 4
        for f in frames:
            assert f.exists() and f.stat().st_size > 0


def test_extract_frames_never_uses_a_rejected_option():
    """Static assertion: the extraction command must not USE -vsync when the
    installed ffmpeg rejects it. Guards against the option creeping back.

    The source may still MENTION "-vsync" as the legacy fallback (that is
    required for old ffmpeg builds), so this asserts on the SELECTION logic
    rather than on the string: the modern flag must be preferred whenever the
    installed ffmpeg accepts it.
    """
    if not shutil.which(media_utils.FFMPEG):
        pytest.skip("ffmpeg not on PATH")
    if not _installed_ffmpeg_supports(["-fps_mode", "vfr", "-f", "null", "-"]):
        pytest.skip("installed ffmpeg supports neither pacing flag")
    assert media_utils._ffmpeg_supports("fps_mode") is True
    # And the helper must correctly report the legacy option as unsupported
    # on this ffmpeg 9 build.
    assert media_utils._ffmpeg_supports("vsync") is False, (
        "_ffmpeg_supports('vsync') must be False on ffmpeg 9+, which removed "
        "the option"
    )
