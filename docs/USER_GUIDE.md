# User & Operator Guide

How to talk to VeriSafe, what you'll get back, and how operators read the audit trail. Written for non-technical users first; the machinery section is at the bottom.

## For users: just message your file or link

Send **anything** to the WhatsApp number — a link, a photo, an audio note, a video, a scanned document, or a downloaded file. You do **not** need to know what kind of thing it is; VeriSafe identifies it automatically.

You will get back, in your own language, three things:

1. **A plain-language answer** — good / caution / do-not-use / can't-tell.
   - *Good*: "This looks genuine. I found no signs of tampering or fraud."
   - *Caution*: "Some details look unusual. Do not pay money or share personal information until you confirm with the official source yourself."
   - *Do-not-use*: "This appears harmful/fake. Do not open it, do not click the link, do not send it further."
   - *Can't-tell*: "I couldn't verify this fully (explain why). Treat it as untrusted until you confirm another way."
2. **A confidence statement** — how sure we are, phrased plainly ("high confidence", "moderate", "low"), so you can weigh it. A cautious user reads a low-confidence 'good' the same as a caution.
3. **Practical next steps** — what to do with the result (ignore / verify with official source / delete and block sender).

### Language support
VeriSafe auto-detects which language you wrote in and replies in the same one. Currently: **English, Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali.** If detection fails it falls back to English.

### What to never do
- Don't expect a yes/no on legal/medical/financial decisions — this is a technical-signal tool, not a judge.
- Don't treat "looks genuine" as proof a government ID belongs to a specific person; it checks *tampering and format*, not *identity*. Confirm identity through the issuing authority.
- Anything marked *do-not-use* for a downloaded file: don't open it on any device, especially if it came from an unknown sender.

## For operators: reading the audit trail

Every job appends one record to `VERISAFE_AUDIT_LOG`:

```
[HH:MM:SS] job=<id> target=<url_phishing|malicious_file|gov_document|deepfake_video|...> \
          verdict=<trust|caution|do_not_use|unable_to_verify> conf=<0..1> \
          wall_s=<s> short_circuited_at=<cap|null> stage_timings={...}
```

Fields that matter during an incident:

- **`conf`** — calibrated probability after reliability scaling. Low + `unable_to_verify` means coverage was thin (deps missing / media poor / budget hit), not that the input is safe.
- **`stage_timings`** — per-stage wall seconds (P8). If one stage dominates, that's where to look for thermal/CPU pressure.
- **`short_circuited_at`** — the capability whose *confirmed* positive triggered the conservative early-stop; later heavy stages show as `skip_early_stop`. Presence here means the finding was strong enough that skipping more analysis bought CPU headroom without changing the answer.

The full structured evidence (every `CheckResult`) is in `JobOutcome.to_dict()` if you log beyond the summary line.

### Quick operator runbook (this laptop)
- Before a bulk/batch run: check `tail ~/.hermes/logs/thermal_power_monitor.log`; keep `VERISAFE_FFMPEG_THREADS=2`; don't overlap other Hermes compute.
- Watch the quarantine mount size (`df -h $(echo $VERISAFE_QUARANTINE)`). If it grows, the stale-sweep may have failed — the TTL is `VERISAFE_STALE_TTL_S`.
- To force a clean re-deploy: restart OpenWA, then VeriSafe; webhook self-re-registers on boot.
