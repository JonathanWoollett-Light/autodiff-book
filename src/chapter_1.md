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

# Forward Automatic Differentiation

With some function $ f(x_1,x_2,...,x_n)=y_1,y_2,...,y_m $ the Big O'notation of automatic differentiation is $ O(n) $ for forward autodiff and $ O(m) $ for reverse autodiff.

With forward differentiation we need a run for each input, while for reverse differentiation we need a run for each output.

## Setup

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

## Forward

In forward differentiation with each pass through our function we calculate one of the partial deriatives for our inputs.

Our function outputs an `e` value and a deriative ($\frac{\delta e}{\delta a}$ or $\frac{\delta e}{\delta b}$). We use a seed value for each input to determine for which input we are calculating the partial derivative. Our function signature becomes:

```rust
fn our_function(a: f32, b: f32, a_: f32 b_: f32) -> (f32,f32)
```

With each expression we multiply the previous derivatives effectively suming them.

For deriative accumulation at each stage we multiply the cumulative derivative by the deriative of the expression (the affect it has relative to the given variable).

For clarification underlined variales like `x_` will be repsented like $ x_d $.

If we perform a run to obtain $\frac{\delta e}{\delta a}$ it follows:

### Step 1

Given:

$$ \frac{\delta c}{\delta a} = b, \frac{\delta c}{\delta b} = a $$

$$ \therefore c_d =  \frac{\delta c}{\delta b} a_d + \frac{\delta c}{\delta b} b_d $$

To propagate the derivative we care about ($ \frac{\delta c}{\delta a} $) we multiply by our seeds values `a_` and `b_` to propagate the specific deriative forward.
```rust
let c_ = a * b_ + b * a_;
```

### Step 2

$$ \frac{\delta d}{\delta a} = \cos(a) $$

$$ \therefore d_d = \cos(a) a_d $$

```rust
let d_ = a.cos() * a_;
```

### Step 3

$$ \frac{\delta e}{\delta c} = 1, \frac{\delta e}{\delta d} = 1 $$

$$ \therefore \frac{\delta e}{\delta a} = \frac{\delta e}{\delta c} c_d + \frac{\delta e}{\delta d} d_d $$

```rust
let e_ = 1 * c_ + 1 * d_; // `let e_ = c_ + d_;`
```

### Summary

$$ \frac{\delta e}{\delta a} = \frac{\delta e}{\delta c} \cdot (\frac{\delta c}{\delta a} a_d + \frac{\delta c}{\delta b} b_d)+ \frac{\delta e}{\delta d} \cdot (\frac{\delta d}{\delta a} a_d)$$

```rust
let e_ = 1*(b*a_ + a*b_) + 1*(a.cos() * a_); // = 1*c_ + 1*d_;
```

- When $a_d=1$ (`a_=1`) and $b_d=0$ (`b_=0`):

  $ \frac{\delta e}{\delta a} = \frac{\delta e}{\delta c} \cdot \frac{\delta c}{\delta a}+ \frac{\delta e}{\delta d} \cdot \frac{\delta d}{\delta a} $

  ```rust
  let e_ = 1*b + 1*a.cos();
  ```

- When $a_d=0$ (`a_=0`) and $b_d=1$ (`b_=1`):

  $ \frac{\delta e}{\delta a} = \frac{\delta e}{\delta c} \cdot \frac{\delta c}{\delta b} $

  ```rust
  let e_ = 1*a;
  ```

Our final function being:
```rust
fn our_autodiff_function(a: f32, b: f32, a_: f32, b_: f32) -> (f32,f32) {
    let c = a * b;
    let c_ = a*b_ + b*a_;
    let d = a.sin();
    let d_ = a.cos() * a_;
    let e = c + d;
    let e_ = c_ + d_;
    return (e, e_);
}
```

For calculating both derivatives $\frac{\delta e}{\delta a}$ and $\frac{\delta e}{\delta b}$:
```rust
fn outer(a: f32, b: f32) -> (f32,f32) {
    let a_d = our_autodiff_function(a,b,1.,0.);
    let b_d = our_autodiff_function(a,b,0.,1.);
    (a_d,b_d)
}
```
### Optimisation

We can avoid re-calculating the same intermediary values each run by splitting `our_autodiff_function` into 2:

```rust
fn forward(a: f32, b: f32) -> (f32,f32,f32) {
    let c = a * b;
    let d = a.sin();
    let e = c + d;
    return (c,d,e);
}
fn backward(a: f32, b: f32, c: f32,d: f32,e: f32,a_,b_) -> f32 {
    let c_ = a*b_ + b*a_;
    let d_ = a.cos() * a_;
    let e_ = c_ + d_;
    return e_;
}
fn outer(a: f32, b: f32) -> (f32,f32) {
    let (c,d,e) = forward(a,b);
    let a_d = backward(a,b,c,d,e,1.,0.);
    let b_d = backward(a,b,c,d,e,0.,1.);
    (a_d,b_d)
}
```

In this optimisation we have lost something however, now we need to hold the values of `a`, `b`, `c`, `d` and `e` in memory for the entire lifetime of our `outer` function. This could potentially be very expensive.

To avoid this and maintain our speedup we could rewrite it:

```rust
fn forward_autodiff(a: f32, b: f32) -> (f32,f32,f32) {
    let c = a * b;
    let (c_a, c_b) = (c_(a, b, 1., 0.), c_(a, b, 0., 1.));
    // `b` can be dropped here
    let d = a.sin();
    let (d_a, d_b) = (d_(a, 1.), d_(a, 0.));
    // `a` can be dropped here
    let e = c + d;
    // `c` and `d` can be dropped here
    let (e_a, e_b) = (e_(c_a,d_a), e_(c_b,d_b));
    return (e, e_a, e_b);
}
fn c_(a: f32, b: f32, a_: f32, b_: f32) -> f32 { a*b_ + b*a_ }
fn d_(a: f32, a_:f32) -> f32 { a.cos() * a_ }
fn e_(c_: f32, d_: f32) -> f32 { c_ + d_ }

fn outer(a: f32, b: f32) -> (f32,f32) {
    let (e, a_d, b_d) = forward_autodiff(a, b);
    (a_d, b_d)
}
```
Now we only calculate `c`, `d` and `e` once while dropping variables as to minimise memory usage.

While we have somewhat obfuscated the O'notation, at each derivative step we still need to calculate $n$ deriatives, so our function is still $O(n)$.

Further optimization is possible, although this is beyound what this introductory guide will cover.