# QR Verification Design — offline govt-ID QR checks

*Hermes Agent (ox-alpha) for DawnofGenX — 2026-08-22*

Feeds the `src/verisafe/qr_verify/` implementation. Scouting evidence and
source links: `docs/research/QR_VERIFY_SCOUTING_2026-08-22.md`.

## Adoption decisions (from scouting)

- **Decoder:** `zxing-cpp` primary; `cv2.QRCodeDetector` fallback. pyzbar
  rejected (libzbar0 system dep, stale). wechat_qrcode deferred until real
  photo scans prove insufficient.
- **Aadhaar Secure QR:** own decode + RSA-SHA256 signature verification
  against pinned UIDAI public certificates in
  `src/verisafe/assets/uidai_certs/` (no third-party runtime dep).
- **EPIC (Voter ID):** static-key AES/CBC decrypt per captn3m0/epicqr spec.
- **PAN:** legacy plain-text parsing only. PAN "Enhanced 2.0 Secure QR" is
  reverse-engineered/unofficial — display-only, best-effort.
- **EPFO UAN:** format undocumented — display-only until a real card sample
  exists.
- **Compliance:** Aadhaar Act §29 analysis from the scouting report holds
  under VeriSafe's zero-retention design (masked-UID-only output, transient
  in-memory processing, user consent). Not legal advice.

## Pipeline

```
image (path | PIL | ndarray)
      |
      v
[decoder]  zxing-cpp -> cv2.QRCodeDetector fallback
      |
      v  payload: str
[classifier]  aadhaar_secure | epic_b64 | digilocker_url | pan_text | unknown
      |
      v
[per-kind verifier]
   aadhaar_secure.py  int->bytes->gzip->[0xFF fields][256B RSA sig]
                      verify RSA-SHA256 vs pinned UIDAI certs (+ test keys)
   epic.py            base64->AES/CBC(static key/iv)->JSON {epic_no,...}
   pan_legacy.py      regex PAN extraction + name-line heuristics
   url                host sanity only
      |
      v
QrVerifyResult(kind, status ok|degraded|failed|unavailable,
               signals{...}, detail)
```

## What each document type proves

| Type | Cryptographic | Structural only | Unknown/display-only |
|---|---|---|---|
| Aadhaar Secure QR | UIDAI RSA-SHA256 signature over demographics | field presence/format | |
| EPIC | — | AES decrypts to spec JSON | |
| PAN legacy card | — | PAN regex + text fields | |
| PAN Enhanced 2.0 / e-PAN | | | unofficial decoders only |
| EPFO UAN | | | no public spec sampled |
| DigiLocker/UDYAM URL | | | deep-link host check |

**Honesty boundaries (what QR verification does NOT prove):**
possession of a card ≠ ownership of the identity it names; a photo of a
genuine card can be replayed by anyone holding it; PAN-2.0 decoding relies on
unofficial reverse engineering; UAN format is undocumented. Signals feed the
fusion layer as *evidence*, never as an automatic verdict.

## Trust-store model

- Certificates are UIDAI-*published public keys* (X.509 PEM despite `.cer`
  extension), mirrored at adoption time from the MIT-licensed reference repo;
  provenance + SHA-256 pins recorded in `src/verisafe/assets/uidai_certs/README.md`.
- Rotation procedure: re-download from uidai.gov.in's certificate page when
  UIDAI publishes new keys, drop-in replace, update README hashes. No code
  changes required (verifier iterates all certs in the directory).
- Test trust anchors ride via `extra_trust_paths` only — never committed into
  the production cert directory.

## Test strategy

- Hermetic synthetic fixtures: deterministic test-signature keypair mints a
  golden Aadhaar-style numeric payload (private key exists only inside the
  fixture generator, never in-repo); `qrcode` lib renders images; numpy
  noise + Gaussian blur emulate phone-photo degradation.
- Tamper tests flip one body byte → signature must fail.
- Golden EPIC vector (`ABC1234566`) from the epicqr SPEC exercises AES path.
- Real-card smoke tests: pending operator-supplied cards (2–3), run manually,
  never committed.
