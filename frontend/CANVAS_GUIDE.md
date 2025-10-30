Canvas usage guide for Orbimesh frontend

What is <canvas>?
- The HTML5 <canvas> element provides a drawing surface in which you can render pixels directly using JavaScript (2D or WebGL). It's different from DOM or SVG: canvas is immediate-mode and pixel-based.

When to use canvas (why)
- Use canvas when you need high-performance pixel rendering: games, particle systems, complex charts with thousands of shapes, freehand drawing, image editing, or WebGL-powered 3D scenes.
- Prefer DOM elements (HTML) or SVG when you need accessibility, readable content, selectable text, or when styling/interaction with individual elements is required.

How to use canvas (best practices)
- Keep drawing loop separate and use requestAnimationFrame for animations.
- Handle devicePixelRatio: set canvas.width/height to CSS size * DPR and use ctx.setTransform to scale.
- Cleanup: on component unmount remove animation frames and event listeners to avoid leaks.
- Security: when embedding arbitrary HTML/JS into iframes, use sandbox attributes. Don’t directly inject scripts into the host page.
- Accessibility: provide a textual fallback or representational content for screen readers.

When NOT to use canvas
- For rich text, forms, or where SEO and accessibility matter, prefer HTML/SVG.

How Orbimesh uses canvas
- The backend may generate HTML canvas-based content and place it in conversationState.canvas_content with canvas_type 'html'. Frontend displays it inside a sandboxed iframe (see `orchestration-details-sidebar.tsx` and `interactive-chat-interface.tsx`).
- The iframe uses `sandbox="allow-scripts allow-same-origin"` so scripts inside can run but are contained from the parent page.

Troubleshooting checklist
- If canvas content looks blurry: ensure devicePixelRatio handling is applied in the generated HTML (scale canvas accordingly).
- If embedded scripts don't run: check that the iframe isn't missing sandbox permissions or that srcDoc content includes the script correctly.
- If canvas never appears: confirm backend sets `has_canvas` and `canvas_content` in conversation state; frontend listens and renders only when both are present.

Quick examples
- Use DOM for counters and simple UI (example: `/public/counter-demo.html` contains both a DOM counter and a small canvas animation).
- See `frontend/components/orchestration-details-sidebar.tsx` and `frontend/components/interactive-chat-interface.tsx` for how canvas content is displayed (iframe for HTML or markdown rendering otherwise).

If you want, I can:
- Run a focused audit on canvas-related code paths and implement fixes (resize, cleanup, sanitization).
- Add automatic tests or eslint rules to prevent insecure iframe usage.
- Generate standardized HTML wrapper that backend can use to ensure DPR scaling and safe embedding.

Let me know which of the above you want me to do next. If you'd like me to run or build the frontend/backend, tell me and I will provide the exact commands for your Windows PowerShell shell so you can run them.