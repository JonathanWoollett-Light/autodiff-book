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

# Reverse Automatic Differentiation

With some function $ f(x_1,x_2,...,x_n)=y_1,y_2,...,y_m $ the Big O'notation of reverse automatic differentiation is $ O(m) $.

With reverse differentiation we need a run for each output.

Our function signature becomes:

```rust
fn our_function(a: f32, b: f32, e_: f32) -> (f32,f32,f32)
```

Unlike forward differentiation we need to seperate the affects of variables in seperate expressions into seperate variables. If we have `let x = a + 2;` and `let y = 2*a` we need seperate quantities to represent the seperate affects of `a` upon these expressions $\delta y_a $ and $\delta x_a $.

For easy cloning of variables we will use a macro `dup!(x,2)` which returns a tuple with the given number of elements of clones of the given variable.

## Example run

If we perform a run through our [Setup](./chapter_0.md) function to obtain $\frac{\partial e}{\partial a}$ and $\frac{\partial e}{\partial b}$ it follows:

### Step 1
$$ 
    \frac{\partial e}{\partial c} = 1
    ,\qquad
    \frac{\partial e}{\partial d} = 1 
$$
$$ \therefore 
    \delta e_d = \frac{\partial e}{\partial c} \cdot \delta e
    ,\qquad
    \delta e_c = \frac{\partial e}{\partial d} \cdot \delta e 
$$
$$ \therefore 
    \delta e_c = \delta e 
    ,\qquad 
    \delta e_d = \delta e 
$$
```rust
let (e_c,e_d) = dup!(e_, 2);
```
### Step 2
The quantitiy describing the affect of $ d $ on the output of the function is the sum of all quantities $ x_d $ each which describe the affects of $ d $ on expressions which sum into the output. In this case there is only one $ e_d $.

$$ \delta d = \delta e_d $$
```rust
let d_ = e_d;
```

### Step 3
$$ \frac{\partial d}{\partial a} = \cos(a) $$
$$ \therefore 
    \delta a_d = \frac{\partial d}{\partial a} \cdot \delta d
$$
$$ \therefore 
    \delta d_a = \cos(a) \cdot \delta d
$$
```rust
let (d_a,) = (a.cos() * d_,);
```

### Step 5
$$ \delta c = \delta e_c $$
```rust
let c_ = e_c;
```

### Step 6
$$ 
    \frac{\partial c}{\partial a} = b
    ,\qquad
    \frac{\partial c}{\partial b} = a 
$$
$$ \therefore 
    \delta c_a = \frac{\partial c}{\partial a} \cdot \delta c
    ,\qquad
    \delta c_b = \frac{\partial c}{\partial b} \cdot \delta c
$$
$$ \therefore 
    \delta c_a = b \cdot \delta c 
    ,\qquad 
    \delta c_b = a \cdot \delta c
$$
```rust
let (c_a,c_b) = (b*c_, a*c_);
```
### Step 7

At this point we sum all the duplicates of our inputs.

$$ 
    \delta a = \delta d_a + \delta c_a
    ,\qquad 
    \delta b = \delta c_b
$$
```rust
let (a_,b_) = (d_a+c_a, c_b);
```

### Summary

Our final function being:
```rust
fn our_autodiff_function(a: f32, b: f32, e_: f32) -> (f32,f32,f32) {
    // Forward
    let c = a * b;
    let d = a.sin();
    let e = c + d;
    // Backward
    let (e_c, e_d) = dup!(e_, 2);
    let d_ = e_d;
    let (d_a,) = (a.cos() * d_,);
    let c_ = e_c;
    let (c_a,c_b) = (b*c_, a*c_);
    let (a_,b_) = (d_a+c_a, c_b);

    return (e, a_, b_);
}
```

As you can see in this code unlike forward auto-diff some of the variables are used multiple times in the backward section (in this specific case only the `a` variable is duplicated).

In application these implicit copies may be undesireable, as such this can be replace with manual cloning using `dup!` to make the code more explicit. You could for instance return a tuple of shared pointers ([`Rc`](https://doc.rust-lang.org/std/rc/struct.Rc.html)s or [`Arc`](https://doc.rust-lang.org/std/sync/struct.Arc.html)s) which may allow dropping the object earlier and conserving memory.