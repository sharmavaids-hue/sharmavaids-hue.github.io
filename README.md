# Aryan Sharma — Portfolio

Personal site for Aryan Sharma (Computer Science & Mathematics, University of Maryland).
Plain HTML, CSS and JavaScript — no build step — so it deploys straight to GitHub Pages.

## Structure

```
index.html      page content
styles.css      design system (paper / ink / vermillion; Newsreader, DM Sans, JetBrains Mono)
main.js         three small interactive simulations + scroll reveals
assets/         portrait, project screenshots, résumé PDF
```

## The simulations (all vanilla JS, canvas 2D)

| Where | What | Math |
|---|---|---|
| Hero strip | Heat equation — drag to add heat | ∂u/∂t = α∇²u, explicit finite differences, insulated boundaries |
| Experience | Gradient descent with momentum — click to drop an optimizer | v ← βv − η∇f on a non-convex surface; contours via marching squares |
| Projects | Monte Carlo — click to resimulate | Geometric Brownian motion, P5–P95 band, terminal histogram |

Each simulation pauses when off-screen and respects `prefers-reduced-motion`.

## Editing

- **Text:** everything lives in `index.html`, one section per comment block.
- **Résumé:** replace `assets/Aryan_Sharma_Resume.pdf` (keep the filename).
- **Colors / fonts:** the `:root` variables at the top of `styles.css`.

## Deploy (GitHub Pages)

Settings → Pages → Source: *Deploy from a branch* → `main` / `(root)`.
