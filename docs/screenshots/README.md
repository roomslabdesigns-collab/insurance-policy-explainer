# Screenshots

Not yet captured in this session — browser automation wasn't available. Rather than fabricate placeholder images, here's exactly what to capture and how, so this folder can be filled in accurately in a couple of minutes.

## How

1. `run_app.bat` (or start the server + `streamlit run app.py` manually), then open http://localhost:8501.
2. Upload `data/policies/sample_health_policy.pdf` (already included, synthetic and safe to show publicly).
3. Take each screenshot below (Win+Shift+S on Windows), save as the listed filename in this folder, PNG, ~1200-1600px wide is plenty — no need for full 4K screenshots in a repo.

## Shots needed

| # | Filename | What to capture |
|---|---|---|
| 1 | `01_upload.png` | The sidebar after uploading the sample policy and clicking "Process Policy" — showing the Policy Information panel (name, version, chunk count, "Ready"). |
| 2 | `02_supported_answer.png` | Ask *"Is there a waiting period for pre-existing conditions?"* — the resulting `Covered` answer card with its evidence-quality badge and citation line. |
| 3 | `03_citation.png` | The same answer card, cropped to show the "Direct Policy Text" excerpt + "Source: Section → Clause → Page" line clearly. |
| 4 | `04_highlighted_evidence.png` | Click "🔍 View Evidence in Policy" on that answer — the rendered PDF page with the supporting sentence highlighted in yellow. |
| 5 | `05_abstention.png` | Ask *"Does this policy cover space travel?"* — the safe abstention response (note it should NOT look like an error — that's the point). |
| 6 | `06_dashboard.png` | `run_dashboard.bat` → http://localhost:8502 — the Quality/Safety metric tiles plus the classification breakdown chart. |

## Then, in `README.md`

Replace the `docs/screenshots/` link text with an inline markdown image for each, e.g.:

```markdown
### Uploading a policy
![Policy upload and processing](docs/screenshots/01_upload.png)
```
