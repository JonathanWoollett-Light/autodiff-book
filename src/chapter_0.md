<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css" integrity="sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.js"                  integrity="sha384-K3vbOmF2BtaVai+Qk37uypf7VrgBubhQreNQe9aGsz9lB63dIFiQVlJbr92dw2Lx" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/contrib/auto-render.min.js"    integrity="sha384-kmZOZB5ObwgQnS/DuDg6TScgOiWWBiVt0plIRkZCmE6rDZGrEOQeHM5PcHi+nyqe" crossorigin="anonymous"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "\\(", right: "\\)", display: false},
                {left: "$", right: "$", display: false},
                {left: "\\[", right: "\\]", display: true}
            ]
        });
    });
</script>

# Setup

The chain rules gives:
$$\frac{dy}{dx}=\frac{dy}{dz} \cdot \frac{dz}{dx} $$
We have a function:
```rust
fn our_function(a: f32, b: f32) -> f32 {
    let c = a * b;
    let d = a.sin();
    let e = c + d;
    return e;
}
```
We want to calculate $ \frac{\delta e}{\delta a} $ and $ \frac{\delta e}{\delta b} $.

In the mathematical formulas underlined variables like `x_` will be repsented like $ \delta x $ or variables like `x_a` represented like $ \delta x_a $.