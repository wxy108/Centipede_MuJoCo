# Centipede Backbone Wave: Amplitude, Quarter Wavelength, and Leg Circle Overlap

## 1. Setup and Notation

| Symbol | Definition |
|--------|-----------|
| $a$ | Link length (distance between adjacent joints) |
| $w$ | Half body width |
| $\ell$ | Leg projected length (radius of reachability circle) |
| $N$ | Number of segments |
| $n$ | Number of body waves along the chain |
| $A$ | Joint angle amplitude (rad) |
| $k$ | Spatial wavenumber: $k = \frac{2n\pi}{(N-1)a}$ |
| $\varepsilon$ | Small-bending parameter: $\varepsilon = \frac{A}{ak}$ |
| $s$ | Arc-length coordinate along the backbone |
| $u$ | Phase variable: $u = ks$ |

The joint angle at segment $j$ is:

$$\alpha_j = A \sin\!\Bigl(-\frac{2n\pi}{N-1}\, j\Bigr)$$

The continuum heading field is:

$$\varphi(s) = \varepsilon \cos(ks)$$

---

## 2. Backbone Curve: Parametric Form up to Third Order

### 2.1 x-component (even orders only)

**First order** $O(1)$:

$$x^{(1)}(s) = s$$

**Second order** $O(\varepsilon^2)$:

$$x^{(2)}(s) = \Bigl(1 - \frac{\varepsilon^2}{4}\Bigr)\,s \;-\; \frac{\varepsilon^2}{8k}\,\sin(2ks)$$

**Third order**: No $O(\varepsilon^3)$ correction to $x$ (cosine expansion produces only even powers).

$$x^{(3)}(s) = x^{(2)}(s)$$

### 2.2 y-component (odd orders only)

**First order** $O(\varepsilon)$:

$$y^{(1)}(s) = \frac{\varepsilon}{k}\,\sin(ks)$$

**Second order**: No $O(\varepsilon^2)$ correction to $y$ (sine expansion produces only odd powers).

$$y^{(2)}(s) = y^{(1)}(s)$$

**Third order** $O(\varepsilon^3)$:

$$y^{(3)}(s) = \frac{\varepsilon}{k}\Bigl(1 - \frac{\varepsilon^2}{8}\Bigr)\sin(ks) \;-\; \frac{\varepsilon^3}{72k}\,\sin(3ks)$$

---

## 3. Amplitude

### 3.1 First-order amplitude

$$\boxed{B^{(1)} = \frac{|\varepsilon|}{k} = \frac{A}{ak^2} = \frac{A(N-1)^2 a}{4n^2\pi^2}}$$

### 3.2 Third-order amplitude

The $y$-curve at third order is:

$$y^{(3)}(s) = B_1\sin(ks) - B_3\sin(3ks)$$

where the fundamental and third-harmonic amplitudes are:

$$B_1 = \frac{|\varepsilon|}{k}\Bigl(1 - \frac{\varepsilon^2}{8}\Bigr), \qquad B_3 = \frac{|\varepsilon|^3}{72k}$$

The peak value of $y^{(3)}$ occurs where $\frac{dy}{ds} = 0$:

$$B_1\cos(u) - 3B_3\cos(3u) = 0$$

For small $\varepsilon$, $B_3 \ll B_1$, so the peak is near $u = \pi/2$. A perturbation expansion gives the **third-order amplitude**:

$$\boxed{B^{(3)} \approx B_1 + B_3 = \frac{|\varepsilon|}{k}\Bigl(1 - \frac{\varepsilon^2}{8}\Bigr) + \frac{|\varepsilon|^3}{72k}}$$

More precisely, the maximum satisfies:

$$B^{(3)} = B_1 - B_3 \cdot \bigl(-\sin(3u^*)/\sin(u^*)\bigr)$$

evaluated at the critical point $u^*$. For practical purposes, the leading correction to amplitude relative to first order is:

$$B^{(3)} \approx B^{(1)}\Bigl(1 - \frac{\varepsilon^2}{8} + \frac{\varepsilon^2}{72}\Bigr) = B^{(1)}\Bigl(1 - \frac{8\varepsilon^2}{72}\Bigr) = B^{(1)}\Bigl(1 - \frac{\varepsilon^2}{9}\Bigr)$$

### 3.3 Amplitude ratio

$$\frac{B^{(3)}}{B^{(1)}} \approx 1 - \frac{\varepsilon^2}{9}$$

The amplitude shrinks slightly at higher order due to the nonlinear projection effect.

---

## 4. Quarter Wavelength

### 4.1 Definition

The **quarter wavelength** $\lambda/4$ is defined as the $x$-distance from a peak of $y$ to the nearest zero crossing. This is the spatial extent of one "leg group" (half of the segments between a peak and a trough lie within $\lambda/4$).

### 4.2 First-order quarter wavelength

At first order, $x \approx s$ and $y \approx B^{(1)}\sin(ks)$, so peaks occur at $ks = \pi/2 + m\pi$ and zero crossings at $ks = m\pi$.

$$\boxed{\frac{\lambda^{(1)}}{4} = \frac{\pi}{2k} = \frac{(N-1)a}{4n}}$$

### 4.3 Third-order quarter wavelength

At third order, the mapping $s \to x$ introduces compression:

$$x(s) = \Bigl(1 - \frac{\varepsilon^2}{4}\Bigr)s - \frac{\varepsilon^2}{8k}\sin(2ks)$$

The **mean projected wavelength** (ignoring the $\sin(2ks)$ ripple) is:

$$\lambda_{\text{eff}} = \Bigl(1 - \frac{\varepsilon^2}{4}\Bigr)\frac{2\pi}{k}$$

Therefore:

$$\boxed{\frac{\lambda^{(3)}}{4} \approx \Bigl(1 - \frac{\varepsilon^2}{4}\Bigr)\frac{\pi}{2k} = \frac{\lambda^{(1)}}{4}\Bigl(1 - \frac{\varepsilon^2}{4}\Bigr)}$$

Additionally, the $\sin(2ks)$ ripple in $x(s)$ introduces a small oscillation in the apparent zero-crossing positions. At a zero crossing of $y$ (where $ks = m\pi$), the ripple term $\sin(2m\pi) = 0$, so the zero-crossing positions in $x$ are unaffected by the ripple. At a peak of $y$ (where $ks = \pi/2 + m\pi$), the ripple term $\sin(\pi + 2m\pi) = 0$ as well. Therefore:

> The $\sin(2ks)$ ripple does **not** shift peak or zero-crossing positions, and the quarter wavelength formula above is exact to $O(\varepsilon^2)$.

### 4.4 Quarter wavelength ratio

$$\frac{\lambda^{(3)}/4}{\lambda^{(1)}/4} = 1 - \frac{\varepsilon^2}{4}$$

---

## 5. Circle Overlap Geometry for Leg Clustering

### 5.1 Problem Statement

At each segment $i$, the body-width attachment points (left/right) are located at:

$$\mathbf{p}_{\text{left},i} = \mathbf{p}_{\text{mid},i} + w\,\hat{\mathbf{n}}_i, \qquad \mathbf{p}_{\text{right},i} = \mathbf{p}_{\text{mid},i} - w\,\hat{\mathbf{n}}_i$$

where $\hat{\mathbf{n}}_i$ is the unit normal to the backbone at segment $i$, and $\mathbf{p}_{\text{mid},i}$ is the midpoint of segment $i$.

Each leg can reach any point within a circle of radius $\ell$ centered at its attachment point. The **reachability circle** for leg $i$ (left side) is:

$$\mathcal{C}_i = \bigl\{\mathbf{p} : \|\mathbf{p} - \mathbf{p}_{\text{left},i}\| \leq \ell\bigr\}$$

### 5.2 Pairwise Overlap Condition

Two circles $\mathcal{C}_i$ and $\mathcal{C}_j$ overlap if and only if:

$$d_{ij} = \|\mathbf{p}_{\text{left},i} - \mathbf{p}_{\text{left},j}\| < 2\ell$$

When they overlap, the intersection area is:

$$A_{\text{overlap}} = 2\ell^2 \arccos\!\Bigl(\frac{d_{ij}}{2\ell}\Bigr) - \frac{d_{ij}}{2}\sqrt{4\ell^2 - d_{ij}^2}$$

### 5.3 Critical Pair: Peak vs Quarter-Wave

Within a leg group (half wavelength), the **most separated** pair on one side is the segment at the **peak** and the segment at the **quarter-wave point** (zero crossing of the backbone). These two have the largest distance $d_{\max}$ among all pairs in the group.

The distance between their left-side attachment points determines whether full-group clustering is geometrically possible:

$$d_{\text{peak-QW}} = \|\mathbf{p}_{\text{left,peak}} - \mathbf{p}_{\text{left,QW}}\|$$

**Clustering feasibility criterion:**

$$\boxed{d_{\text{peak-QW}} < 2\ell \quad \Longleftrightarrow \quad \text{all feet in group can share a common point}}$$

### 5.4 Estimating $d_{\text{peak-QW}}$

Using the third-order backbone, the peak is at $s_p$ where $ks_p \approx \pi/2$, and the quarter-wave point is at $s_0$ where $ks_0 \approx 0$ (or $\pi$).

The backbone separation between these points is approximately $(\Delta x, \Delta y) \approx (\lambda/4, \; B)$ (one quarter wave horizontally, one full amplitude vertically).

The body-width normals point in different directions at peak vs zero-crossing:
- At the **peak**: the backbone is locally flat (zero curvature), normal points straight up/down
- At the **zero crossing**: the backbone has maximum curvature, normal is tilted

The attachment point separation is thus:

$$d_{\text{peak-QW}} \approx \sqrt{\Bigl(\frac{\lambda}{4}\Bigr)^2 + B^2 + w^2\bigl(2 - 2\cos\Delta\theta\bigr) + \text{cross terms}}$$

where $\Delta\theta$ is the heading difference between peak and zero-crossing segments. A full analytical expression requires the specific heading values, so numerical evaluation is preferred.

### 5.5 Common Overlap Region

The **common overlap** of all circles in a group is:

$$\mathcal{R} = \bigcap_{i \in \text{group}} \mathcal{C}_i$$

If $\mathcal{R} \neq \emptyset$, there exists a single point where all feet can land. The centroid of $\mathcal{R}$ provides the **optimal foot target point**, and the leg angle for segment $i$ to reach this target is:

$$\theta_{\text{leg},i} = \arctan\!\Bigl(\frac{y_{\text{target}} - y_{\text{left},i}}{x_{\text{target}} - x_{\text{left},i}}\Bigr)$$

---

## 6. Summary of Key Formulas

| Quantity | 1st Order | 3rd Order |
|----------|-----------|-----------|
| Backbone $y$ | $\frac{\varepsilon}{k}\sin(ks)$ | $\frac{\varepsilon}{k}(1-\frac{\varepsilon^2}{8})\sin(ks) - \frac{\varepsilon^3}{72k}\sin(3ks)$ |
| Backbone $x$ | $s$ | $(1-\frac{\varepsilon^2}{4})s - \frac{\varepsilon^2}{8k}\sin(2ks)$ |
| Amplitude $B$ | $\frac{\|\varepsilon\|}{k}$ | $\frac{\|\varepsilon\|}{k}(1-\frac{\varepsilon^2}{8}) + \frac{\|\varepsilon\|^3}{72k}$ |
| Quarter wavelength $\lambda/4$ | $\frac{\pi}{2k}$ | $(1-\frac{\varepsilon^2}{4})\frac{\pi}{2k}$ |
| Leg clustering feasible | — | $d_{\text{peak-QW}} < 2\ell$ |

where $\varepsilon = A/(ak)$, $k = 2n\pi/[(N-1)a]$.

---

## 7. Practical Notes

1. **When is the small-bending expansion accurate?** When $|\varepsilon| \lesssim 0.3$. For the default parameters ($A = 0.3$, $n = 1.5$, $N = 20$), $\varepsilon \approx 0.32$, which is borderline — the 3rd order correction matters.

2. **Rotation to center on x-axis:** The discrete backbone has a slight drift angle $\delta$ due to finite-segment effects. Rotate by $-\delta$ where $\delta = \arctan(y_N - y_0,\; x_N - x_0)$ is the chord angle from first to last joint.

3. **Group definition:** One leg group contains all segments between consecutive zero crossings of $y$ on the centered backbone. For $n$ body waves, there are approximately $2n$ groups (alternating left/right clustering). Each group has approximately $(N-1)/(2n)$ segments.

4. **Overlap depends on $\ell/(\lambda/4)$ ratio:** Clustering is easier when legs are long relative to the quarter wavelength. Increasing $n$ (more waves) or decreasing $A$ both reduce the quarter wavelength and amplitude, making overlap more likely.
