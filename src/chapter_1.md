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

With some function $ f(x_1,x_2,...,x_n)=y_1,y_2,...,y_m $ the Big O'notation of forward automatic differentiation is $ O(n) $.

With reverse differentiation we need a run for each input.

In the simplest implementation with forward differentiation with each pass through our function we calculate the partial deriatives for one of our inputs (e.g. $\frac{\delta y_1}{\delta x_4}$,$\frac{\delta y_2}{\delta x_4}$,...,$\frac{\delta y_m}{\delta x_4}$).

In our specific case we have a function which has a single output (`e`), therefore we are concerned with computing $\frac{\delta e}{\delta a}$ and $\frac{\delta e}{\delta b}$.

For this simple implementation that caluclate the partial derivatives for 1 input in each run, we need to pass a set of seeds which effectively determine which input we are calculating the partial deratives for. In implementation this is multiple `f32`s where 1 is `1.` for the input we are calculating and the rest are `0.`.

E.g.

```rust
fn our_function(a: f32, b: f32, a_: f32 b_: f32) -> (f32,f32)
```

In forward automatic differenation we accumulative the deriatives of input/s effectively summing their effects. To do this in each operation we multiply the cumulative derivative of an input for a variable by the deriative of the expression with respect to the aforementioned variable.

Consider the statements:

```rust
let a = x+2.;
// `x` has a constant affect on `a` and thus the derivative with respect to `x` is 
//  `1.`.
// We set a variable `x_a` equal to `1.`.
let b = a*3.;
// `a` has a linear affect on `b` and thus the derivative with respect to `a` is `3.`, 
//  but we don't care about the derivative of this internal variable.
// So now we want to get the cumulative derivative that accumulates the affects of `x` 
//  through these statements.
// Simply here we multiply the derivative of `b` with respect to `a` by the cumualtive 
//  deriative of `x` for `a` and thus we get the affect of `x` on `b`  (`db/dx`).
// We set a variable `x_b` equal to `3*x_a` (e.g. `3.*1.`) 
let c = x/2.;
// We do much the same as we did with `a`.
// We set a variable `x_c` equal to `0.5`.
let d = c+c.
// We do much the same as we did with `b`.
// d/dc(c+c)=2, the deriative of d with respect to c is `2.`.
// We set a variable `x_d` equal to `2.*x_c` (e.g. `2.*1.`)
let e = b+d;
// In this case we are affectively joining the 2 paths of `x` so far.
// Mathematicaly we would do `d/db(b+d)` and `d/dd(b+d)` with each multiplying the respective 
//  accumulative derivative for `x`, e.g. `1.*x_b` and `1.*x_d`.
// We set a variable `x_e` equal to `1.*x_b + 1.*d+x`
```

It may help your understanding to see that mathematically for each statement we are getting the derivative for **every** previous variable, e.g. for `let e = b + d` we are getting all the deriatives $\frac{\delta e}{\delta x}$, $\frac{\delta e}{\delta a}$, $\frac{\delta e}{\delta b}$, $\frac{\delta e}{\delta c}$ and $\frac{\delta e}{\delta d}$, and then multiplying all of these by their respective cumulative deriatives for the inputs $\delta x$, $\delta x_a$, $\delta x_b$, $\delta x_c$ and $\delta x_d$ (`x_`, `x_a`, `x_b`, `x_c` and `x_d`). But since all of these deriatives except the one which have components present in the statements work out to be 0, it means:

$$ \frac{\delta e}{\delta x} \cdot \delta x + \frac{\delta e}{\delta a} \cdot \delta x_a + \frac{\delta e}{\delta b} \cdot \delta x_b + \frac{\delta e}{\delta c} \cdot \delta x_c + \frac{\delta e}{\delta d} \cdot \delta x_d \equiv \frac{\delta e}{\delta b} \cdot \delta x_b + \frac{\delta e}{\delta d} \cdot \delta x_d $$

Since:

$$ \frac{\delta e}{\delta x} \cdot \delta x + \frac{\delta e}{\delta a} \cdot \delta x_a + \frac{\delta e}{\delta b} \cdot \delta x_b + \frac{\delta e}{\delta c} \cdot \delta x_c + \frac{\delta e}{\delta d} \cdot \delta x_d \equiv 0 \cdot \delta x + 0 \cdot \delta x_a + 1 \cdot \delta x_b + 0 \cdot \delta x_c + 1 \cdot \delta x_d $$

Thus while for any operation we are effectively summing all the derivative by all the cumulative derivatives, since we have specific knowledge about operations we optimize this by suming only the deriatives for which we know are useful (non-zero).

Understanding this helps you understand how this can be applied to any operation and automatic differentiation is not simply formed from some random rules.


## Example run

If we perform a run to obtain $\frac{\delta e}{\delta a}$ on the function given in [Setup](./chapter_0.md) it follows:

### Step 1

$$ c = a \cdot b $$
$$ \therefore \frac{\delta c}{\delta a} = b, \frac{\delta c}{\delta b} = a $$
$$ \therefore \delta a_c = \frac{\delta c}{\delta a} \delta a + \frac{\delta c}{\delta b} \delta b $$
$$ \therefore \delta a_c = b \cdot \delta a + a \cdot \delta b $$

```rust
let a_c = b * a_ + a * b_;
```

### Step 2


$$ d = \sin(a) $$
$$ \therefore \frac{\delta d}{\delta a} = \cos(a) $$
$$ \therefore \delta a_d = \cos(a) \cdot \delta a $$

```rust
let a_d = a.cos() * a_;
```

### Step 3

$$ e = c + d $$
$$ \therefore \frac{\delta e}{\delta c} = 1, \frac{\delta e}{\delta d} = 1 $$
$$ \therefore \delta a_e = \frac{\delta e}{\delta c} \cdot \delta a_c + \frac{\delta e}{\delta d} \cdot \delta a_d $$
$$ \therefore \delta a_e = 1 \delta a_c + 1 \delta a_d $$

```rust
let a_e = 1. * a_c + 1. * a_d;
```

### Summary

```rust
let a_e = 1.*(b*a_ + a*b_) + 1.*(a.cos() * a_);
```

- When $\delta a=1$ (`a_=1`), $\delta b=0$ (`b_=0`):

  ```rust
  let a_e = b + a.cos();
  ```

- When $a_d=0$ (`a_=0`) and $b_d=1$ (`b_=1`):

  ```rust
  let b_e = a;
  ```

Therefore:

$$ \frac{\delta e}{\delta a} = b + \cos(a), \frac{\delta e}{\delta b} = a $$

Our final function being:
```rust
fn our_autodiff_function(a: f32, b: f32, a_: f32, b_: f32) -> (f32,f32) {
    let c = a * b;
    let c_ = a*b_ + b*a_; // This being `a_c` in our `a` specific example.
    let d = a.sin();
    let d_ = a.cos() * a_; // This being `a_d` in our `a` specific example.
    let e = c + d;
    let e_ = c_ + d_; // This being `a_e` in our `a` specific example.
    return (e, e_);
}
```

For calculating both derivatives $\frac{\delta e}{\delta a}$ and $\frac{\delta e}{\delta b}$:
```rust
fn outer(a: f32, b: f32) -> (f32,f32) {
    let a_e = our_autodiff_function(a,b,1.,0.);
    let b_e = our_autodiff_function(a,b,0.,1.);
    (a_e,b_e)
}
```

If we wanted to minimise the `e` we simply subtract a proportion of `a_e` and `b_e` from our input `a` and `b` values. Doing this is gradient descent.

### Optimisation

We can avoid re-calculating the intermediary values each run by splitting `our_autodiff_function` into 2 functions:

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
fn outer(a: f32, b: f32) -> (f32,f32,f32) {
    let (c,d,e) = forward(a,b);
    let a_d = backward(a,b,c,d,e,1.,0.);
    let b_d = backward(a,b,c,d,e,0.,1.);
    (e,a_d,b_d)
}
```

In this optimisation we have lost something however, now we need to hold the values of `a`, `b`, `c`, `d` and `e` in memory for the entire lifetime of our `outer` function. This could potentially be very expensive.

To avoid this and maintain our speedup we could rewrite it:

```rust
fn forward_autodiff(a: f32, b: f32) -> (f32,f32,f32) {
    let c = a * b;
    let (a_c, b_c) = (c_(a, b, 1., 0.), c_(a, b, 0., 1.));
    // `b` can be dropped here
    let d = a.sin();
    let (a_d, b_d) = (d_(a, 1.), d_(a, 0.));
    // `a` can be dropped here
    let e = c + d;
    // `c` and `d` can be dropped here
    let (a_e, b_e) = (e_(a_c,a_d), e_(b_c,b_d));
    return (e, a_e, b_e);
}
fn c_(a: f32, b: f32, a_: f32, b_: f32) -> f32 { a*b_ + b*a_ }
fn d_(a: f32, a_:f32) -> f32 { a.cos() * a_ }
fn e_(c_: f32, d_: f32) -> f32 { c_ + d_ }

fn outer(a: f32, b: f32) -> (f32,f32,f32) {
    let (e, a_e, b_e) = forward_autodiff(a, b);
    (e,a_e, b_e)
}
```
Now we only calculate `c`, `d` and `e` once and drop the variables asap to minimise memory usage.

While we have somewhat obfuscated the O'notation, at each derivative step we still need to calculate $n$ deriatives, so our function is still $O(n)$.

Further optimization is possible, although this is beyound what this introductory guide will cover.