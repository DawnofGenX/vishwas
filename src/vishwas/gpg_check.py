"""GPG keyring-backed digital-signature verification (D2).

Operator-managed truststore model:
  - VISHWAS_GPG_TRUSTSTORE (default ~/.vishwas/gpg-truststore) holds public
    key blobs (.asc/.gpg) and/or an ownertrust.txt.
  - Per run, keys are imported into a THROWAWAY GNUPGHOME under the job's
    quarantine root (zero retention — deleted with the job).
  - A signature from a NON-imported key is 'untrusted', not an error.

All gpg interaction is subprocess-based with hardened flags
(--batch --no-tty --status-fd 1), hard timeouts, and parsing of [GNUPG:]
status lines ONLY. Never raises. Zero network.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TRUSTSTORE = Path.home() / ".vishwas" / "gpg-truststore"
_TIMEOUT_S = 60


@dataclass
class GpgVerdict:
    valid: bool | None          # True/False = cryptographic result; None = unverifiable
    fingerprint: str | None     # signer key fingerprint (hex, no spaces)
    trusted: bool | None        # True = signer in configured truststore
    err: str | None             # short machine-readable error class/message


def _gpg_bin() -> str | None:
    return shutil.which(os.environ.get("VISHWAS_GPG_BIN", "gpg") or "gpg")


def available() -> bool:
    """True iff a gpg binary is resolvable."""
    return _gpg_bin() is not None


def _run_gpg(gpg: str, homedir: Path, args: list[str], timeout: int = _TIMEOUT_S) -> tuple[int, str]:
    """Run gpg against an explicit homedir; returns (returncode, status-fd text).
    Never raises — transport failures become (1, '[GNUPG:] ERRSIG ...')-style
    synthetic lines so callers can treat them uniformly."""
    cmd = [gpg, "--homedir", str(homedir), "--batch", "--no-tty",
           "--status-fd", "1", *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 1, "[GNUPG:] ERRSIG timeout\n"
    except Exception as e:  # noqa: BLE001
        return 1, f"[GNUPG:] ERRSIG {type(e).__name__}\n"


def _status_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("[GNUPG:]")]


class GpgKeyring:
    """A throwaway GNUPGHOME with operator keys imported for one job run."""

    def __init__(self, homedir: Path):
        self.homedir = Path(homedir)
        self._bin = _gpg_bin()
        self._init_ok = False
        if self._bin is not None:
            try:
                self.homedir.mkdir(parents=True, exist_ok=True)
                os.chmod(self.homedir, 0o700)
                rc, _out = _run_gpg(self._bin, self.homedir,
                                     ["--list-keys"], timeout=30)
                # any rc is fine here — the point is the homedir now exists
                self._init_ok = True
            except Exception:  # noqa: BLE001
                self._init_ok = False

    @property
    def usable(self) -> bool:
        return self._init_ok and self._bin is not None

    def add_key(self, blob: bytes | Path) -> bool:
        """Import one public-key blob. Returns True on success."""
        if not self.usable:
            return False
        gpg = self._bin
        assert gpg is not None  # narrowed by usable property
        try:
            if isinstance(blob, Path):
                src = str(blob)
            else:
                tmp = self.homedir / "_import_tmp.asc"
                tmp.write_bytes(blob)
                src = str(tmp)
            rc, out = _run_gpg(gpg, self.homedir, ["--import", src])
            ok = rc == 0 or "imported" in out.lower()
            if isinstance(blob, bytes):
                try:
                    (self.homedir / "_import_tmp.asc").unlink(missing_ok=True)
                except OSError:
                    pass
            return ok
        except Exception:  # noqa: BLE001
            return False

    def set_ownertrust_full(self, fingerprint: str) -> bool:
        """Mark a key's ownertrust as FULL in this throwaway home."""
        if not self.usable:
            return False
        try:
            ot = self.homedir / "ownertrust.txt"
            line = f"{fingerprint.upper()}:6:\n"  # 6 = FULL
            with open(ot, "a") as f:
                f.write(line)
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_fingerprints(self) -> list[str]:
        """Full 40-char fingerprints of PRIMARY keys only in this homedir
        (subkey fprs are excluded — ownertrust is keyed on the primary)."""
        if not self.usable:
            return []
        gpg = self._bin
        assert gpg is not None  # narrowed by usable property
        _rc, out = _run_gpg(gpg, self.homedir, ["--with-colons", "--list-keys"])
        seen: set[str] = set()
        prim: list[str] = []
        last_kind = ""
        for ln in out.splitlines():
            parts = ln.split(":")
            if not parts or len(parts) < 10:
                continue
            if parts[0] in ("pub", "sub", "ssb"):
                last_kind = parts[0]
            elif parts[0] == "fpr" and parts[9]:
                # an 'fpr' record follows the pub/sub line whose key it belongs to;
                # keep only those immediately after a 'pub' (primary key) line
                if last_kind == "pub" and parts[9] not in seen:
                    seen.add(parts[9])
                    prim.append(parts[9])
        return prim

    def verify(self, data_bytes: bytes, sig_bytes_or_path: bytes | Path) -> GpgVerdict:
        """Verify a detached signature over data. Never raises."""
        if not self.usable:
            return GpgVerdict(None, None, None, "gpg binary missing or homedir unusable")
        gpg = self._bin
        assert gpg is not None  # narrowed by usable property
        sig_path = None
        data_path = None
        try:
            data_path = self.homedir / "_verify_data.bin"
            data_path.write_bytes(data_bytes)
            if isinstance(sig_bytes_or_path, Path):
                sig_path = str(sig_bytes_or_path)
            else:
                sp = self.homedir / "_verify_sig.gpg"
                sp.write_bytes(sig_bytes_or_path)
                sig_path = str(sp)
            rc, out = _run_gpg(gpg, self.homedir,
                               ["--verify", sig_path, str(data_path)])
            return _parse_verify(out)
        except Exception as e:  # noqa: BLE001
            return GpgVerdict(None, None, None, f"{type(e).__name__}: {e}")
        finally:
            for pth in (data_path,):
                if pth is not None:
                    try:
                        Path(pth).unlink(missing_ok=True)
                    except OSError:
                        pass
            if sig_path and sig_path.startswith(str(self.homedir)):
                try:
                    Path(sig_path).unlink(missing_ok=True)
                except OSError:
                    pass


def _parse_verify(status_out: str) -> GpgVerdict:
    """Parse [GNUPG:] status lines from gpg --verify output only.

    Handles both upstream and distro-patched GnuPG VALIDSIG layouts. The
    canonical primary fingerprint is taken from the KEY_CONSIDERED line
    (emitted by this build right before/after VALIDSIG); as fallback the
    LAST 40-char-hex token of VALIDSIG (upstream layout puts primary-fpr
    near the end; distro builds may differ). GOODSIG/BADSIG carry only a
    16-char long key-id — used only when no full fpr is available.
    """
    valid: bool | None = None
    fingerprint: str | None = None      # full 40-char preferred
    keyid_16: str | None = None         # short key-id fallback
    err: str | None = None
    considered_fp: str | None = None    # from [GNUPG:] KEY_CONSIDERED <fpr> <n>
    valisig_hexes: list[str] = []       # all 40-hex tokens seen in VALIDSIG
    saw_validsig = False

    for ln in _status_lines(status_out):
        body = ln[9:].strip() if ln.startswith("[GNUPG:]") else ln.strip()
        fields = body.split()
        if not fields:
            continue
        code = fields[0]
        if code == "KEY_CONSIDERED":
            # [GNUPG:] KEY_CONSIDERED <key-or-fpr> <considered-count>
            cand = fields[1].upper() if len(fields) >= 2 else ""
            if len(cand) == 40 and all(c in "0123456789ABCDEF" for c in cand):
                considered_fp = cand
        elif code == "VALIDSIG":
            # Layout varies: collect every 40-hex token; the LAST is the
            # primary-key fingerprint in both observed variants.
            toks = [t.upper() for t in fields
                    if len(t) == 40 and all(c in "0123456789ABCDEF" for c in t)]
            valisig_hexes.extend(toks)
            saw_validsig = True
        elif code in ("GOODSIG", "BADSIG"):
            k = fields[1].upper() if len(fields) >= 2 else ""
            if len(k) == 40 and all(c in "0123456789ABCDEF" for c in k):
                fingerprint = k           # some builds put the full fpr here
            elif len(k) <= 16:
                keyid_16 = k
            valid = True if code == "GOODSIG" else False
            if code == "BADSIG":
                err = "bad signature (digest mismatch)"
        elif code in ("EXPKEYSIG", "EXPUSER"):
            valid = False
            err = "expired signing key"
            if len(fields) >= 2 and not keyid_16:
                keyid_16 = fields[1].upper()
        elif code in ("REVKEYSIG", "REVUSER"):
            valid = False
            err = "revoked signing key"
            if len(fields) >= 2 and not keyid_16:
                keyid_16 = fields[1].upper()
        elif code == "NO_PUBKEY":
            err = "signing key not present in keyring"
        elif code == "ERRSIG":
            err = " ".join(fields[1:])[:120]

    # Resolve the best available identifier:
    # 1) explicit full fpr captured from GOODSIG (rare layout)
    # 2) last 40-hex VALIDSIG token (primary fpr in both observed layouts)
    # 3) KEY_CONSIDERED full fpr — ONLY when VALIDSIG absent (distro builds
    #    may list a subkey fingerprint there; the LAST VALIDSIG hex is safer)
    # 4) 16-char long key-id as a degraded but still identifiable value
    if not saw_validsig and considered_fp:
        valisig_hexes = [considered_fp]
    if valisig_hexes:
        fingerprint = valisig_hexes[-1]
    elif fingerprint is None and keyid_16:
        fingerprint = keyid_16

    if valid is None and err is None:
        err = "no decisive GNUPG status line"
    return GpgVerdict(valid, fingerprint, None, err)


def _blob_primary_fpr(blob) -> str | None:
    """Primary (40-char) fingerprint of a single public-key blob, using an
    isolated throwaway GNUPGHOME. Returns None if unreadable/not a key."""
    if not available():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="vishwas-gpg-fpr-"))
    try:
        kr = GpgKeyring(tmp)
        if not kr.add_key(blob):
            return None
        fps = kr.list_fingerprints()
        return fps[0] if fps else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _glob_key_blobs(dirpath: Path) -> list[Path]:
    if not dirpath.is_dir():
        return []
    return sorted(p for p in dirpath.iterdir()
                  if p.suffix.lower() in (".asc", ".gpg") and p.is_file())


def load_truststore(dirpath: Path | str) -> list[str]:
    """Import every .asc/.gpg blob under dirpath into a throwaway homedir and
    return the resulting PRIMARY-key fingerprints. Tolerates missing/empty
    dirs ([])."""
    blobs = _glob_key_blobs(Path(dirpath))
    out: list[str] = []
    seen: set[str] = set()
    for b in blobs:
        fp = _blob_primary_fpr(b)
        if fp and fp not in seen:
            seen.add(fp)
            out.append(fp)
    return out


def verify_with_truststore(data_bytes: bytes, sig_bytes_or_path: bytes | Path,
                           truststore_dir: Path | str | None = None,
                           untrusted_dir: Path | str | None = None,
                           workdir: Path | None = None) -> GpgVerdict:
    """Zero-network union-keyring verification.

    Builds ONE throwaway GNUPGHOME containing the union of:
      - public keys from ``truststore_dir``   (operator-vouched), and
      - public keys from ``untrusted_dir``    (present but not vouched).
    Only the truststore primaries get ``ownertrust=FULL``; the rest stay at
    their natural (undefined) trust. Because there is no network, any key the
    verifier needs MUST already live in one of those two local rings.

    Result semantics:
      trusted=True   -> valid signature AND signer in the truststore
      trusted=False  -> valid signature but signer only in untrusted ring
                        (or known but not vouched) => caller downgrades
      trusted=None   -> validity undetermined (e.g. NO_PUBKEY, garbled sig)
    Never raises; transport/homdir problems degrade to ``err`` + ``valid=None``.
    """
    ts_dir = Path(truststore_dir) if truststore_dir else DEFAULT_TRUSTSTORE
    ut_dir = (Path(untrusted_dir) if untrusted_dir
              else Path.home() / ".vishwas" / "gpg-known")
    ts_fps = {fp.upper() for fp in load_truststore(ts_dir)}

    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="vishwas-gpg-run-"))
    kr = GpgKeyring(base)
    try:
        if not kr.usable:
            return GpgVerdict(None, None, None, "gpg binary missing or homedir unusable")
        # Import every key from both local rings (dedup is safe/no-op).
        for b in _glob_key_blobs(ts_dir):
            kr.add_key(b)
        for b in _glob_key_blobs(ut_dir):
            kr.add_key(b)
        # Vouch only the truststore primaries.
        for fp in ts_fps:
            kr.set_ownertrust_full(fp)

        v = kr.verify(data_bytes, sig_bytes_or_path)
        if v.valid is True and v.fingerprint:
            v.trusted = v.fingerprint.upper() in ts_fps
        else:
            v.trusted = None
        return v
    except Exception as e:  # noqa: BLE001
        return GpgVerdict(None, None, None, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
