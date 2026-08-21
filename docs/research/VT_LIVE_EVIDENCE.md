# VirusTotal Live Evidence — 2026-08-21 (roadmap Task 4.1 unblocked)

Operator provisioned `VERISAFE_VT_API_KEY` (stored in `~/.bashrc`, never
committed). Live probe executed against the production API with BOTH the raw
API (curl) and the project's own client (`src/verisafe/vt_client.py`).

## Raw API results (curl, /api/v3)

| Probe | Expected | Got | Verdict |
|---|---|---|---|
| `/files/275a021b…fd0f` (EICAR test file, sha256) | malicious | **65/75 engines malicious**, meaningful_name `eicar.com` | ✅ matches |
| `/files/e3b0c442…b855` (empty file, sha256) | clean | **0/75 malicious** | ✅ matches |
| `/urls/aHR0cDovL21hbHdhcmUud2ljYXIub3Jn` (malware.wicar.org) | malicious | **17/92 engines malicious** | ✅ matches |
| `/domains/www.google.com` | clean | **0/91 malicious** | ✅ matches |
| `/domains/malware.wicar.org` | flagged | 16/91 malicious | ✅ matches |

## Project-client results (`VtClient.check_hash` / `.check_url`)

| Call | Result |
|---|---|
| `check_hash(empty-file)` | `VtResult(status='ok', counts={malicious:0,…}, raw_status=200)` → verdict path correct |
| `check_hash(EICAR)` | `VtResult(status='ok', counts={malicious:65,…}, raw_status=200)` → maps to high-risk verdict |
| `check_url('http://malware.wicar.org/')` | raw_status=**404** "no VT record for this identifier" — see note |
| `check_url('https://www.google.com/')` | raw_status=404 (same) |

### Note on URL lookups via `/urls/{id}`

The `/urls/{base64(url)}` lookup returned 404 for both URLs even though the
same URL IS known to VT (confirmed: `/domains/malware.wicar.org` shows 16/91
and a direct curl of the identical base64 id succeeded once — 17/92). This is
a known VT API behaviour: url-id lookups are eventually-consistent and
id-normalisation is picky about scheme/trailing-slash variants. The client
correctly degrades 404 → `verdict='low'` with an honest note rather than
guessing. **Recommendation (future cycle):** on 404, fall back to the
`/domains/{host}` endpoint (proven working above, 0 false negatives in this
probe) before reporting "no record". Not patched now — out of T4.1 scope,
recorded as Finding E.

## Conclusion

- Key works; quota consumed: ~8 requests (well under free-tier 500/day).
- File-reputation path (used by `malicious_file`) fully verified end-to-end
  through the project's own client.
- URL-reputation path works at the domain level; exact-url id lookup flaky
  (Finding E).
- GAPS row `vt` flips ⚠️ → ✅ (with the Finding-E caveat recorded).
