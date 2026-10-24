# CV

`cv.html` is the source of the CV published at
`docs/assets/cv/Abdellatif-Belmady-CV.pdf` (the **Download CV** button of the site).

It uses the same facts as `docs/experience.md` and `docs/education.md`: when a role,
date or certification changes, update the site pages **and** this file, then regenerate the PDF.

## Regenerate the PDF

Headless Chrome is enough (no Node or Python dependency). From the repository root:

```bash
# Windows (Git Bash)
"C:/Program Files/Google/Chrome/Application/chrome.exe" --headless --no-pdf-header-footer \
  --allow-file-access-from-files --virtual-time-budget=10000 \
  --print-to-pdf="docs/assets/cv/Abdellatif-Belmady-CV.pdf" \
  "file:///$(pwd -W)/tools/cv/cv.html"

# macOS / Linux
google-chrome --headless --no-pdf-header-footer --allow-file-access-from-files \
  --virtual-time-budget=10000 \
  --print-to-pdf=docs/assets/cv/Abdellatif-Belmady-CV.pdf "file://$PWD/tools/cv/cv.html"
```

Alternatively open `cv.html` in Chrome → Print → *Save as PDF*, margins **None**,
*Background graphics* on.

The layout is two A4 pages (`.page`): if content is added, check that nothing runs
into the footer line at the bottom of each page.
