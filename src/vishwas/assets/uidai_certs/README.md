# UIDAI offline-verification public certificates

RSA-2048 X.509 certificates published by the Unique Identification Authority
of India (UIDAI) for verifying **offline Aadhaar Secure-QR paper QR codes**.
This directory holds PUBLIC KEYS ONLY — no private key material, no personal
data. Public certificates are safe to redistribute.

## Contents

| File | Subject CN | Notes |
|---|---|---|
| `uidai_offline_publickey_26022021.cer` | DS UNIQUE IDENTIFICATION AUTHORITY OF INDIA 05 | 2021 signing cert |
| `uidai_offline_publickey_2026.cer` | DS Unique Identification Authority of India 06 | current-era signing cert |
| `uidai_offline_publickey_17022026.cer` | DS Unique Identification Authority of India 05 | 2026 signing cert |

Despite the `.cer` extension these files are **PEM-encoded** X.509
certificates (BEGIN/END CERTIFICATE blocks). `vishwas.qr_verify.aadhaar_secure`
loads them with `cryptography.x509.load_pem_x509_certificate` (DER is tried as
a fallback) and extracts the SPKI public key once per process, cached.

## Provenance

Mirrored from the MIT-licensed reference project
`github.com/StarkAg/aadhaar-secure-qr-verifier`, which in turn republishes the
certificates UIDAI distributes with its offline Aadhaar / Secure QR Code
verification material. Format details are documented in the repo's
`docs/research` QR scouting report. Port of method, not copy of code.

## If UIDAI rotates keys (re-pull procedure)

1. Download the new public certificate from UIDAI's official offline-Aadhaar
   pages (uidai.gov.in → Aadhaar Services → Offline Paperless ekyc / Secure
   QR), or from an updated release of the mirror above.
2. Verify the certificate fingerprint against a value UIDAI published
   out-of-band (their app/portal) — never trust an in-band file alone.
3. Drop the `.cer` file into this directory (PEM or DER both load).
4. Sanity-check before committing:

   ```bash
   PYTHONPATH=src python3 - <<'EOF'
   from cryptography import x509
   from pathlib import Path
   for p in sorted(Path("src/vishwas/assets/uidai_certs").glob("*.cer")):
       c = x509.load_pem_x509_certificate(p.read_bytes())
       print(p.name, "->", c.subject.rfc4514_string())
   EOF
   ```

5. Commit. `vishwas.qr_verify` picks up new `.cer` files automatically —
   no code change needed (anchors are discovered by glob at call time).
