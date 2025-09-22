# CVA Model Equivalence Math (HW1F vs G2++)

This note collects the exact mathematics used in the implementation for exposure, CVA, calibration attribution, and bucketed DV01 deltas.

## Market and credit setup
- Discount curve: continuously-compounded zero curve with discount factor \(P(0,t)\). In code we access \(P(0,t)\) via the curve handle.
- Credit: constant hazard intensity \(\lambda\). Survival: \(S(t)=e^{-\lambda t}\). Default increment on bucket \((t_{i-1}, t_i]\): \(\Delta\mathrm{PD}_i = S(t_{i-1})-S(t_i)\).

## Short-rate models and discount bonds
### Hull–White (1F)
- Dynamics under \(\mathbb{Q}\): \(r(t)=x(t)+\phi(t)\), \(\mathrm{d}x=-a x\,\mathrm{d}t+\sigma\,\mathrm{d}W\).
- Define \(h=T-t\). The standard affine form for the discount bond at time t is
\[ P^{\mathrm{HW}}(t,T) = A(t,T)\,\exp\big(-B(t,T)\,x(t)\big). \]
- We use the formula that ties to the initial curve directly as a ratio:
\[ B(t,T)=\frac{1-e^{-a h}}{a},\qquad \operatorname{Var}\Big(\int_t^T x(u)\,\mathrm{d}u\Big)
= \frac{\sigma^2}{a^2}\Big(h - \tfrac{2}{a}(1-e^{-a h}) + \tfrac{1}{2a}(1-e^{-2 a h})\Big). \]
- Then
\[ P^{\mathrm{HW}}(t,T) 
= \frac{P(0,T)}{P(0,t)}\,\exp\Big( -B(t,T)\,x(t) + \tfrac{1}{2}\,\mathrm{Var}\Big(\int_t^T x\Big)\Big). \]
This avoids recomputing \(A(t,T)\) by reusing the market curve.

### G2++ (2F)
- Dynamics: \(r(t)=x(t)+y(t)+\phi(t)\),
\(\mathrm{d}x=-a x\,\mathrm{d}t+\sigma\,\mathrm{d}W_1\),
\(\mathrm{d}y=-b y\,\mathrm{d}t+\eta\,\mathrm{d}W_2\), \(\mathrm{d}W_1\mathrm{d}W_2=\rho\,\mathrm{d}t\).
- Let \(h=T-t\). Define
\[ B_x(t,T)=\frac{1-e^{-a h}}{a},\qquad B_y(t,T)=\frac{1-e^{-b h}}{b}. \]
- Variances and covariance of the integral of the factors on \([t,T]\):
\[ \operatorname{Var}\Big(\int_t^T x\Big) = \frac{\sigma^2}{a^2}\Big(h - \tfrac{2}{a}(1-e^{-a h}) + \tfrac{1}{2a}(1-e^{-2 a h})\Big), \]
\[ \operatorname{Var}\Big(\int_t^T y\Big) = \frac{\eta^2}{b^2}\Big(h - \tfrac{2}{b}(1-e^{-b h}) + \tfrac{1}{2b}(1-e^{-2 b h})\Big), \]
\[ \operatorname{Cov}\Big(\int_t^T x,\int_t^T y\Big) = \frac{\rho\,\sigma\,\eta}{a b}\Big(h - \tfrac{1-e^{-a h}}{a} - \tfrac{1-e^{-b h}}{b} + \tfrac{1-e^{-(a+b)h}}{a+b}\Big). \]
- With \(\operatorname{Var}_{xy}=\operatorname{Var}(\int x)+\operatorname{Var}(\int y)+2\operatorname{Cov}(\int x,\int y)\), the discount bond is
\[ P^{\mathrm{G2}}(t,T) = \frac{P(0,T)}{P(0,t)}\,\exp\Big( -B_x(t,T)\,x(t) - B_y(t,T)\,y(t) + \tfrac{1}{2}\,\operatorname{Var}_{xy} \Big). \]

## Linear instruments and pathwise PVs
- Payer par swap at time t (remaining fixed cashflows \((T_j,\alpha_j)\), fixed rate K, notional N):
  - Remaining annuity: \(A_t = \sum_{T_j>t} \alpha_j P(t,T_j)\)
  - Float PV (single-curve): \(\text{Float}_t = 1 - P(t,T_\mathrm{end})\)
  - Fixed PV: \(\text{Fix}_t = K\,A_t\)
  - Swap PV: \(\mathrm{PV}_t = N\,(\text{Float}_t - \text{Fix}_t)\)
- Par bond at time t (coupon c, accruals \(\alpha_j\), remaining payments at \(T_j\), maturity \(T_m\)):
\[ \mathrm{PV}_t = N\Big( \sum_{T_j>t} c\,\alpha_j P(t,T_j) + P(t,T_m) \Big). \]
- Exposure on a path is \(\max(\mathrm{PV}_t,0)\). EPE at bucket \(t_i\) is the path average of positive MTMs.

## CVA discretization
Given time buckets \(\{t_i\}\):
\[ \mathrm{CVA} \approx \sum_i P(0,t_i)\,\mathrm{EPE}(t_i)\,\Delta\mathrm{PD}_i\,\mathrm{LGD},\quad \Delta\mathrm{PD}_i=S(t_{i-1})-S(t_i). \]
This is the unilateral CVA with independent market/credit and constant LGD.

## Calibration (anchor to ATM level)
- We use a small set of ATM European swaptions \((\text{mat},\text{tenor})\) with a flat Black vol level \(\sigma_{\mathrm{ATM}}\).
- For HW(1F), we fit \(\sigma\) (with \(a\) fixed) by least-squares on swaption prices.
- For G2++, we fit \((\sigma,\eta)\) (with \(a,b,\rho\) fixed) by least-squares on the same helpers.
- In the equivalence demo, we also consider a “level-aligned” G2 with \(\eta\approx 0\) to match HW level risk for linear, level-only books.

## Calibration vs model attribution of CVA difference
Let \(\mathrm{CVA}_{\mathrm{HW}}\) be HW CVA, \(\mathrm{CVA}_{\mathrm{G2}}\) be G2 CVA, and let \(\mathrm{G2}_{\mathrm{aligned}}\) denote G2 with vol pair set to \((\sigma=\sigma_{\mathrm{HW}},\eta\approx 0)\) while keeping \((a,b,\rho)\) fixed and using the same curve.
- Total difference: \(\Delta\mathrm{CVA} = \mathrm{CVA}_{\mathrm{HW}} - \mathrm{CVA}_{\mathrm{G2}}\).
- Calibration component: \(\Delta_{\text{cal}} = \mathrm{CVA}_{\mathrm{G2\,aligned}} - \mathrm{CVA}_{\mathrm{G2}}\).
- Model component: \(\Delta_{\text{model}} = \mathrm{CVA}_{\mathrm{HW}} - \mathrm{CVA}_{\mathrm{G2\,aligned}}\).
- Identity: \(\Delta\mathrm{CVA} = \Delta_{\text{cal}} + \Delta_{\text{model}}\).

## Bucketed DV01-style deltas
We compute CVA sensitivity to a 1 bp shift applied only between two times \((\tau_1,\tau_2]\) under independence assumptions.

### Ratio multiplier (efficient method)
- Under a small parallel shift \(\Delta r\) applied over overlap length \(\ell(t,T)=\max(0,\min(T,\tau_2)-\max(t,\tau_1))\), the base discount ratio is scaled by
\[ \frac{P^{\mathrm{bump}}(t,T)}{P(t,T)} \approx \exp(-\Delta r\,\ell(t,T)). \]
- In our engine, we reuse simulated states \((x(t),y(t))\) and multiply the “base” term \(\frac{P(0,T)}{P(0,t)}\exp(\frac12\mathrm{Var})\) by this factor. For each bucket, we recompute EPE and CVA with the modified base.
- The bucketed delta per 1 bp is \(\Delta\mathrm{CVA}_{\text{bucket}} = \mathrm{CVA}^{\mathrm{bump}} - \mathrm{CVA}.\)

### Brute-force bump (validation)
- Build a bumped discount curve \(\tilde P(0,t)\) such that \(\tilde P(0,t)=P(0,t)\,\exp(-\Delta r\,\max(0,\min(t,\tau_2)-\tau_1))\).
- Recompute CVA with the same simulated states but using the bumped curve to rebuild \(P(t,T)\) via the closed-form ZCB formulas (by replacing \(P(0,\cdot)\) with \(\tilde P(0,\cdot)\)).
- The two methods agree closely; differences are from time discretization and Monte Carlo noise.

## Monte Carlo notes
- Factors are simulated as OU processes with exact discretization. We use antithetic variates and Sobol (optional) to reduce variance.
- Time grid examples: `1M_10Y_3M_30Y` for high resolution near-term and coarser long-end.
- Linear instruments are repriced per path using analytical ZCBs. No lattices.

## Why HW vs G2 equivalence holds here
For a linear, level-driven portfolio, exposure depends primarily on the integral of the short rate. With both models using the same initial curve and ATM level, and with the second factor’s variance set near zero, pathwise ZCBs, forwards, annuities, and therefore EPE and CVA, align within tight tolerances.
