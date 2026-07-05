# Lens4SAE Playground — how to play

## Start
1. Double-click `run_playground.command` (Terminal opens, model loads ~1 min — leave it open).
2. Browser → http://localhost:8765. (Optional public link: also run `run_tunnel.command`.)

## The screen
- **Text box** — type any sentence, hit "Read its mind."
- **Gauge** — the model's sentiment verdict (right/blue = positive, left/red = negative).
- **Chips** — the SAE features ("thoughts") that fired, loudest first. Edge color = screening verdict:
  blue driver-like · green thermometer · yellow indeterminate · gray unscreened.
- **Push panel** — click a chip, set strength, "Push it": the model re-reads with that thought
  amplified; gauge + Δ show whether behavior moved.

## Three experiments
1. **Core lesson:** neutral sentence → push a blue chip (gauge swings) vs a green chip (barely moves).
   Drivers vs thermometers, by hand.
2. **Fight the sentence:** strongly negative text + push a positive driver hard — can an injected
   thought beat the words?
3. **Hunt the gray:** push an unscreened chip at 15 then 30. Growing effect = possible new driver —
   you are literally doing the screening experiment manually.

## Reading a push honestly
- Δ grows with strength → lever (driver-like)
- Δ tiny at any strength → gauge (thermometer)
- Δ shrinks as you push harder → saturation → indeterminate
- Moving the gauge proves the thought does something — not that it is safe to trust.
