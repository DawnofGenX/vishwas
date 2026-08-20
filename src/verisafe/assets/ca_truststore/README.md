# VeriSafe CA Truststore

Deliberately **empty** by default. This directory is scanned at verification
time for X.509 root certificates (`.cer`, `.crt`, `.pem`, `.der`) used to
anchor CMS/PAdES signature chains.

## Policy

- With no anchor matching a signature's issuing CA, VeriSafe reports the
  chain as `incomplete` (tri-state: *unverifiable*, NOT *forged*). An absent
  anchor is never treated as an accusation of fraud.
- Drop in trusted root PEM/DER files only from sources you have deliberately
  decided to trust (e.g. government signing CAs you accept). Malformed
  entries are skipped with a log line and never abort a scan.
- To point a job at an alternate store, set `VERISAFE_CA_TRUSTSTORE=/path/to/dir`.
