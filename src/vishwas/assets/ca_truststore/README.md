# Vishwas CA Truststore

Scanned at verification time for X.509 root certificates (`.cer`, `.crt`,
`.pem`, `.der`) used to anchor CMS/PAdES signature chains
(`vishwas.pades_check.load_trust_store`; default path constant
`_DEFAULT_TRUSTSTORE` in `src/vishwas/capabilities/gov_document.py`,
overridable via `VISHWAS_CA_TRUSTSTORE`).

## Policy

- With no anchor matching a signature's signer, Vishwas reports the chain as
  `incomplete` (tri-state: *unverifiable*, NOT *forged*). An absent anchor is
  never treated as an accusation of fraud.
- `chain == "trusted"` requires the SIGNER certificate itself (subject +
  serial) to be present in this store — i.e. the operator has vouched for the
  exact signing identity, not merely its issuing CA. A store holding only the
  issuing CA still yields `incomplete`.
- Malformed entries are skipped with a log line and never abort a scan.
- To point a job at an alternate store, set `VISHWAS_CA_TRUSTSTORE=/path/to/dir`.

## Anchors

### isrg-root-x1.der — ISRG Root X1

| Field | Value |
|---|---|
| Subject | `C = US, O = Internet Security Research Group, CN = ISRG Root X1` |
| SHA-256 (DER) | `96bcec06264976f37460779acf28c5a7cfe8a3c0aae11a8ffcee05c0bddf08c6` |
| Source bundle | `/etc/ssl/certs/ISRG_Root_X1.pem` (distro CA bundle; PEM SHA-256 `22b557a27055b33606b6559f37703928d3e4ad79f110b407d04986e1843543d1`) |
| Export command | `openssl x509 -in /etc/ssl/certs/ISRG_Root_X1.pem -outform DER -out isrg-root-x1.der` |
| Export date | 2026-08-21 |
| Not after | Jun 4 2035 |

Provenance notes:

- Exported **offline** from the local system trust bundle. No network key
  lookups were performed (offline-only policy). The DER fingerprint above
  equals the publicly documented ISRG Root X1 fingerprint published by ISRG,
  which independently confirms the local bundle copy.
- Purpose: a real, well-formed public root for smoke-testing that the
  production store loads and parses (`test_committed_anchor_smoke`). It does
  NOT anchor any test fixture signature — fixture signers are self-signed test
  identities and are anchored via runtime-generated stores instead.
- Production anchoring is **operator-supplied**: drop in the roots you have
  deliberately decided to trust. No private keys are ever committed here;
  tests generate their own CAs at runtime in temp directories.

## Updating / re-exporting

Re-run the export command against your current distro bundle and update the
table above (fingerprint + date). If you swap in a different root, update
`test_committed_anchor_smoke` in `tests/test_12_pades.py`, which asserts the
documented subject CN.
