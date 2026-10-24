# Portfolio: maintenance notes

Site: <https://abdellatif-belmady.github.io/abdellatif-belmady/>, built with [Zensical](https://zensical.org/).
The repository `README.md` is the GitHub profile page, so these notes live here.

## Layout

| Path | Content |
|---|---|
| `zensical.toml` | Site configuration: navigation, theme, blog, `chatbot_url`, CV path |
| `docs/index.md` | Home page (hero in the front matter, sections in HTML) |
| `docs/experience.md`, `docs/education.md` | Career, education, certifications, awards, languages |
| `docs/blog/posts/` | Blog posts (front matter: `date`, `authors`, `description`, `comments`) |
| `docs/assets/portfolio/` | CSS, assistant script (`js/assistant.js`), logo, favicon, social card |
| `docs/assets/cv/` | CV PDF, generated from `tools/cv/cv.html` (see `tools/cv/README.md`) |
| `theme/` | Template overrides: `home.html` (hero), `main.html` (meta), `partials/` (network SVG, giscus comments) |

## Local preview

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # macOS / Linux: .venv/bin/pip
.venv/Scripts/zensical serve                       # http://localhost:8000
```

## Publishing

Every push to `main` runs `.github/workflows/deploy.yml`: `zensical build --clean`,
then the `site/` folder is published to the `gh-pages` branch that GitHub Pages serves.
The URL does not change.

## Writing a blog post

Create `docs/blog/posts/<slug>.md`:

```markdown
---
date: 2026-10-24
authors: [abdellatif]
description: One sentence for search engines and social cards.
comments: true
---

# Title

Intro paragraph (shown on the blog index).

<!-- more -->

Rest of the post. Cite primary sources at the end.
```

## AI assistant

The chat bubble and the **Explain / Ask** buttons on selected text call the n8n workflow set in
`chatbot_url` (`zensical.toml`). The workflow itself is kept outside the repository.
Leave `chatbot_url = ""` to hide the assistant.
