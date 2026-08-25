"""Robustness transform matrix (media_utils) consistency — P7 red-team focus.

Deepfake detectors trained on clean media fail after WhatsApp-style
transcoding. Vishwas evaluates candidates under a fixed degradation battery.
These tests pin the *composition* of that battery against the real ffmpeg-recipe
source (static, no codec needed), plus one behavioural test proving the
thermal-safe thread cap propagates into every ffmpeg invocation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import vishwas.media_utils as mu

SRC = Path(mu.__file__).read_text()


def _video_matrix_body():
    i = SRC.index("def apply_video_transform_matrix")
    j = SRC.index("def sha256_of", i)
    return SRC[i:j]


def _audio_matrix_body():
    i = SRC.index("def apply_transform_matrix")
    j = SRC.index("def apply_video_transform_matrix", i)
    return SRC[i:j]


# ------------------------------------------------------------ composition ----
def test_video_matrix_covers_required_degradations():
    b = _video_matrix_body()
    names = " ".join(k for k in
                     ("h264_lowbitrate", "hevc_640p", "resize_crop_center",
                      "fps_change_12", "screen_rec_sim", "whatsapp_style"))
    assert "h264" in names and "hevc" in names          # codec axis (H.264 + H.265)
    assert "crop" in names                                # spatial crop axis
    assert "resize" in names or "scale" in b             # spatial resize axis
    assert "fps" in names                                  # temporal FPS-change axis
    assert "screen" in names                               # screen-record simulation
    assert "whatsapp" in names                             # product-relevant transcode
    assert "b:v" in b                                      # explicit bitrate ladders
    # both encoders present
    assert "libx264" in b and "libx265" in b


def test_audio_matrix_covers_whatsapp_voice_note_transcoding():
    b = _audio_matrix_body().lower()
    for req in ("opus", "aac", "mp3", "bandwidth", "noise", "resample"):
        assert req in b, f"audio transform battery missing {req!r}"


# -------------------------------------------------------------- thermal ------
def test_thread_cap_env_tunable_and_applied_to_every_ffmpeg_call(tmp_path, monkeypatch):
    """P8 thermal-safety: every ffmpeg invocation carries -threads <VISHWAS_FFMPEG_THREADS>."""
    recorded: list[list[str]] = []

    class FakeProc:
        returncode = 0

    def fake_run(cmd, **kw):
        recorded.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 4096)     # clears the >2048 existence gate
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    vid = tmp_path / "in.mp4"
    vid.write_bytes(b"x")
    got = mu.apply_video_transform_matrix(vid, tmp_path)

    expected = {"h264_lowbitrate", "hevc_640p", "resize_crop_center",
                "fps_change_12", "screen_rec_sim", "whatsapp_style", "original"}
    assert set(got.keys()) == expected, f"matrix returned {sorted(got)}"

    ff_cmds = [c for c in recorded if c[:1] == [mu.FFMPEG]]
    assert len(ff_cmds) == 6, "one ffmpeg call per generated transform expected"
    for c in ff_cmds:
        assert "-threads" in c, f"ffmpeg cmd lacks thermal thread cap: {c[:6]}"
        idx = c.index("-threads")
        assert c[idx + 1] == str(mu.THREADS)
