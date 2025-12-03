import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_synthetic1(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset with no pieces
    POLYNOMIAL
    """
    def f1(x): 
        return 4*x[0]**4 - 3*x[0]**3 - 5*x[0]**2 + 4*x[0] + 3
    
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 3))
    y_clean = f1(x.T)
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise/100) * std_dev, size=n)
    mask = [np.repeat(True, n)]
    return x, y_noisy, mask, mask


def load_synthetic2(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset with a single threshold-based piecewise function and a slight class imbalance
    The functions are simple
    ABSOLUTE
    """
    def f2(x):
        if x[0] < 0.8:
            return 2 * x[0] + 1
        else:
            return -3 * x[0] + 5
        
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class 1 has', np.sum(x[:, 0] < 0.8), 'samples, and class 2 has', np.sum(x[:, 0] >= 0.8), 'samples') if verbose else None
    y_clean = np.array([f2(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask = np.where(x[:, 0] < 0.8, True, False) 
    mask = [mask, np.logical_not(mask)]
    return x, y_noisy, mask, mask

    
def load_synthetic3(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset with a single threshold-based piecewise function and a slight class imbalance
    The functions are harder
    CURVED
    """
    def f3(x): 
        if x[0] < -1: 
            return x[0]**2 
        else: 
            return x[0]**4 - 3*x[0]**3 + 2*x[0]**2 - 4
    
    np.random.seed(seed)    
    x = np.random.uniform(-3, 3, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class 1 has', np.sum(x[:, 0] < -1), 'samples, and class 2 has', np.sum(x[:, 0] >= -1), 'samples') if verbose else None
    y_clean = np.array([f3(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask = np.where(x[:, 0] < -1, True, False)
    mask = [mask, np.logical_not(mask)]
    return x, y_noisy, mask, mask


def load_synthetic4(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset with a triple threshold-based piecewise function representing tax owed based on income
    Since the values are large, the dataset has to be scaled 
    TAX
    """
    def f4(x): 
        if x[0] < 500: 
            return 0.1 * x[0]
        elif x[0] < 1000:
            # return 5000 + 0.2 * (x[0] - 50000)
            return 50 + 0.3 * (x[0] - 500)
        elif x[0] < 1500:
            # return 15000 + 0.3 * (x[0] - 100000)
            return 200 + 0.55 * (x[0] - 1000)
        else:
            # return 30000 + 0.5 * (x[0] - 150000)
            return 475 + 0.8 * (x[0] - 1500)

        
    np.random.seed(seed)
    x = np.random.uniform(100, 2000, size=(n, 2))
    x[:, 1] = x[:, 1] * 0.1
    print('Class distribution: ', np.sum(x[:, 0] < 500), np.sum((x[:, 0] >= 500) & (x[:, 0] < 1000)), np.sum((x[:, 0] >= 1000) & (x[:, 0] < 1500)), np.sum(x[:, 0] >= 1500)) if verbose else None
    y_clean = np.array([f4(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)

    mask_1 = np.where(x[:, 0] < 500, True, False)
    mask_2 = np.logical_and(x[:, 0] >= 500, x[:, 0] < 1000)
    mask_3 = np.logical_and(x[:, 0] >= 1000, x[:, 0] < 1500)
    mask_4 = np.where(x[:, 0] >= 1500, True, False)
    mask = [mask_1, mask_2, mask_3, mask_4]
    
    # Scale X,y 
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y_noisy = scaler.fit_transform(y_noisy.reshape(-1, 1)).flatten()
    return x, y_noisy, mask, mask


def load_synthetic5(n=600, seed=0, noise=0, verbose=False): 
    """
    Synthetic dataset with a double threshold-based piecewise function 
    Harder conditions that do not depend solely on x[0]
    Easy function
    SEPARATED
    """

    def f6(x): 
        if 2*x[1]**2 - x[2] < 4: 
            return 0.8*x[0]
        else: 
            return -0.2*x[0] + 0.1*x[1]
        
    np.random.seed(seed)
    x = np.random.uniform(0, 3, size=(n, 3))
    print('Class distribution: ', np.sum(2*x[:, 1]**2 - x[:, 2] < 4), np.sum(2*x[:, 1]**2 - x[:, 2] >= 4)) if verbose else None
    y_clean = np.array([f6(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    mask_1 = np.where(2*x[:, 1]**2 - x[:, 2] < 4, True, False)
    mask_2 = np.logical_not(mask_1)
    mask = [mask_1, mask_2]
    return x, y_noisy, mask, mask

def load_synthetic6(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset for 8-bit signed addition with latent overflow label in X, with optional noise on the modulo sum.
    
    - Inputs:
        n     : number of samples
        seed  : random seed
        noise : standard deviation of Gaussian noise added to sum_mod (in least significant bit units)
    - Features (X):
        X[:,0]: operand a (signed 8-bit: -128 to 127)
        X[:,1]: operand b (signed 8-bit: -128 to 127)
        X[:,2]: overflow flag (0=no overflow, 1=overflow)
    - Target (y):
        sum_mod_noisy = (a + b) mod 256, reinterpreted as signed 8-bit, plus Gaussian noise
    """
    np.random.seed(seed)
    low, high = -128, 127
    
    # sample operands
    a = np.random.randint(low, high+1, size=n)
    b = np.random.randint(low, high+1, size=n)
    
    # compute true sum and modulo sum
    true_sum = a + b
    sum_mod_unsigned = np.mod(true_sum, 256)
    sum_mod = ((sum_mod_unsigned + 128) % 256) - 128
    
    # overflow flag (latent in features)
    overflow = ((true_sum < low) | (true_sum > high)).astype(int)
    
    # add Gaussian noise to sum_mod
    if noise > 0:
        # noise interpreted in same units as sum_mod (integer LSBs)
        sum_mod = sum_mod.astype(float)
        sum_mod += np.random.normal(0, noise, size=n)
        # round back to integer and clip to signed 8-bit range
        sum_mod = np.round(sum_mod).astype(int)
        sum_mod = np.clip(sum_mod, low, high)
    
    # assemble feature matrix X and target y
    X = np.column_stack([a, b, overflow])
    y = sum_mod

    # masks 
    mask_no_overflow = overflow == 0
    mask_overflow = overflow == 1
    mask_overflow_plus = np.logical_and(overflow == 1, true_sum > high)
    mask_overflow_minus = np.logical_and(overflow == 1, true_sum < low)
    masks = [mask_no_overflow, mask_overflow_plus, mask_overflow_minus]

    print("Mask counts (no overflow, overflow):") if verbose else None
    print([mask.sum() for mask in masks]) if verbose else None

    X = X.astype(float)
    y = y.astype(float)

    return X, y, masks, masks



def load_synthetic7(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset simulating retail sales with different regimes on weekdays versus weekends.
    - x[:,0] represents the day of the week (0 to 6, with <5 considered weekdays, >=5 as weekend).
    - x[:,1] is the promotion intensity (0 to 1).
    The functions now use quadratic adjustments instead of sine and cosine.
    SALES
    """
    np.random.seed(seed)
    # Simulate day-of-week as integers from 0 to 6.
    day = np.random.randint(0, 7, size=(n, 1)).astype(float)
    # Simulate promotion intensity between 0 and 1.
    promo = np.random.uniform(0, 1, size=(n, 1))
    x = np.hstack([day, promo])
    
    def f8(x):
        # High promotion intensity, doesn't really matter the day
        if x[1] > 0.85: 
            return 200 - 2 * (1.2*x[0] - 6)**2 + 10 * x[1]
        # x[0]: day, x[1]: promotion intensity
        if x[0] < 5:  # Weekdays            
            # Quadratic bump: (day-2.5)^2 gives a bump around the middle of the week.
            return 100 + 10 * x[0] + 20 * x[1] - 0.5 * (x[0] - 2.5)**2
        else:         # Weekends
            # Different linear and quadratic effect for weekend days.
            return 150 + 5 * x[0] + 15 * x[1] + 0.4 * (x[0] - 5)**2
    
    y_clean = np.array([f8(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)

    print('Class distribution: ', np.sum(x[:, 1] > 0.85), np.sum((x[:, 0] < 5) & (x[:, 1] <= 0.85)), np.sum((x[:, 0] >= 5) & (x[:, 1] <= 0.85))) if verbose else None
    
    # Create masks for weekdays and weekends
    mask_weekday = (x[:, 0] < 5) & (x[:, 1] <= 0.85)
    mask_weekend = (x[:, 0] >= 5) & (x[:, 1] <= 0.85)
    mask_high_promo = (x[:, 1] > 0.85)  # High promotion intensity
    mask = np.array([mask_weekday, mask_weekend, mask_high_promo])
    
    return x, y_noisy, mask, mask


def load_synthetic8(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset simulating housing prices based on geographic location.
    - x contains two features: normalized latitude and longitude.
    - The condition is defined by the Euclidean distance from the city center (0.5, 0.5).
      * Urban: distance < 0.3.
      * Suburban: distance >= 0.3.
    Adjustments now use quadratic terms instead of sine/cosine.
    HOUSING
    """
    np.random.seed(seed)
    x = np.random.uniform(0, 1, size=(n, 2))
    center = np.array([0.5, 0.5])
    
    def urban_price(x):
        dist = np.linalg.norm(x - center)
        return 50 * (1 - 2 * dist + 1.5 * dist**2)
    
    def suburban_price(x):
        dist = np.linalg.norm(x - center)
        return 30 * (1 - (dist - 0.3)) + 5 * (dist - 0.3)**2
    
    def f9(x):
        dist = np.linalg.norm(x - center)
        if dist < 0.3:
            return urban_price(x)
        else:
            return suburban_price(x)
    
    y_clean = np.array([f9(xi) for xi in x])
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    
    mask_urban = np.array([np.linalg.norm(xi - center) < 0.3 for xi in x])
    mask_suburban = np.logical_not(mask_urban)
    mask = [mask_urban, mask_suburban]

    print('Class distribution: ', np.sum(mask_urban), np.sum(mask_suburban)) if verbose else None

    # Scale the y values to be between 0 and 1 for better interpretability.
    y_noisy = (y_noisy - np.min(y_noisy)) / (np.max(y_noisy) - np.min(y_noisy))    
    return x, y_noisy, mask, mask 


def load_synthetic9(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset simulating an insurance risk score in a smooth, real-world fashion.
    
    Features:
      - x[:,0]: Age (range: 18 to 70)
      - x[:,1]: Driving experience in years (range: 0 to 50)
      - x[:,2]: Driving risk indicator (range: 0 to 1)
      
    The true risk score is a weighted blend of four regime-specific functions,
    with weights computed via logistic functions of age so the overall function is smooth.
    
    The four underlying regimes (which can be interpreted later) are:
      * Regime 1: Young drivers (age < ~25)
      * Regime 2: Young-to-middle-aged (age ~25 to 40)
      * Regime 3: Middle-aged drivers (age ~40 to 55)
      * Regime 4: Older drivers (age > ~55)
      
    Each regime has its own risk function (a different linear relationship) and the overall output
    is given by a smoothly weighted sum of these functions.
    INSURANCE
    """
    np.random.seed(seed)
    
    # Generate features.
    age = np.random.uniform(16, 70, n)
    experience = np.random.uniform(0, 50, n)
    driving_risk = np.random.uniform(0, 1, n)
    x = np.column_stack([age, experience, driving_risk])
    
    # Define logistic weighting functions based on age to produce smooth transitions.
    # The sharper the denominator (here 3), the more distinct the regions; we use modest slopes for smoothness.
    w1 = 1 / (1 + np.exp((age - 25) / 3))                             # High for age < ~25.
    w2 = 1 / (1 + np.exp((age - 40) / 3)) - 1 / (1 + np.exp((age - 25) / 3))  # Between ~25 and ~40.
    w3 = 1 / (1 + np.exp((age - 55) / 3)) - 1 / (1 + np.exp((age - 40) / 3))  # Between ~40 and ~55.
    w4 = 1 - 1 / (1 + np.exp((age - 55) / 3))                         # High for age > ~55.
    
    # Define regime-specific risk functions.
    # Here, coefficients are chosen for illustration so that each regime has a slightly different slope relative to the inputs.
    f1 = -0.2 * age + 0.30 * experience + 25 * driving_risk      # Regime 1: Young drivers.
    f2 = -0.1 * age + 0.20 * experience + 20 * driving_risk     # Regime 2: Transitioning to middle age.
    f3 = 0.2 * age + 0.15 * experience + 15 * driving_risk     # Regime 3: Middle-aged drivers.
    f4 = 0.7 * age + 0.05 * experience + 10 * driving_risk     # Regime 4: Older drivers.
    
    # The overall output is the smooth blend of the four regimes.
    y_clean = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
    
    std_dev = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise / 100) * std_dev, size=n)
    
    # For analysis purposes, one may segment the data post hoc using simple age thresholds.
    # Here we provide four masks corresponding to approximate segmentation boundaries.
    mask1 = age < 25
    mask2 = (age >= 25) & (age < 40)
    mask3 = (age >= 40) & (age < 55)
    mask4 = age >= 55
    mask = [mask1, mask2, mask3, mask4]

    print('Class distribution: ', np.sum(mask1), np.sum(mask2), np.sum(mask3), np.sum(mask4)) if verbose else None
    
    return x, y_noisy, mask, mask

def load_synthetic10(n=600, seed=0, noise=0, verbose=False):
    """
    Synthetic dataset simulating bug risk estimation for software modules.

    Features (normalized to [0,1]):
      - x[:,0]: LOC
      - x[:,1]: Complexity
      - x[:,2]: Churn

    Piecewise risk:
      Piece 1: ((LOC > 0.5 AND Complexity > 0.7) OR (Churn > 0.8))
      Piece 2: (LOC < 0.3 AND Churn < 0.5)
      Piece 3: else
    BUG
    """
    import numpy as np
    np.random.seed(seed)

    # generate features
    x = np.random.uniform(0, 1, size=(n, 3))

    # scoring function
    def risk_score(xi):
        loc, comp, churn = xi
        if (loc > 0.5 and comp > 0.7) or (churn > 0.8):
            return 5*loc + 3*comp + 2*churn + 1
        elif loc < 0.3 and churn < 0.5:
            return 1*loc + 2*comp + 1.5*churn
        else:
            return 3*loc + 2.5*comp + 2*churn + 5

    # compute outputs
    y_clean = np.array([risk_score(xi) for xi in x])
    std = np.std(y_clean)
    y_noisy = y_clean + np.random.normal(0, (noise/100)*std, size=n)

    # define exclusive condition masks
    c1 = (x[:,0] > 0.5) & (x[:,1] > 0.7)            # LOC>0.7 & Comp>0.7
    c2 = (~c1) & (x[:,2] > 0.8)                     # Churn>0.8 but not c1
    c3 = (~c1 & ~c2) & (x[:,0] < 0.3) & (x[:,2] < 0.5)  # LOC<0.3 & Churn<0.5, not c1/c2
    c4 = ~(c1 | c2 | c3)                            # everything else

    # segment masks follow the piece definitions
    mask1 = c1 | c2      # piece 1
    mask2 = c3           # piece 2
    mask3 = c4           # piece 3

    masks = [mask1, mask2, mask3]
    condition_masks = [c1, c2, c3, c4]

    print("Segment counts:", [m.sum() for m in masks]) if verbose else None              # sums to 500 
    print("Condition counts:", [c.sum() for c in condition_masks]) if verbose else None  # sums to 500

    return x, y_noisy, masks, condition_masks


def load_synthetic11(n=600, *, seed=0, noise=0.0, verbose=False):
    """
    Synthetic crop-yield dataset whose five classes contain (almost) the
    same number of samples.

    The first four classes are the agronomic regimes described in the
    original `load_synthetic11`; any sample that satisfies none of them
    is assigned to the fallback class ‘other’.  
    Samples are generated **per class** until the target counts are met,
    making the final masks mutually exclusive, collectively exhaustive
    and roughly balanced.

    Returns
    -------
    X         : ndarray (n, 3)      – features (temp, rain, sun)
    y_noisy   : ndarray (n,)        – response with additive Gaussian noise
    masks     : list[np.ndarray]    – 5 Boolean masks, one per class
    one_hot   : ndarray (n, 5)      – same information in one-hot form
    """
    import numpy as np

    rng   = np.random.default_rng(seed)
    per_c = [n // 5] * 5                      # target size of each class
    for k in range(n % 5):                    # distribute the remainder
        per_c[k] += 1

    # ------------------------------------------------------------------
    # 1 regime predicates (same formulas as before)
    # ------------------------------------------------------------------
    def predicates(t, r, s):
        c1 = np.sin(t * np.pi / 30) + np.log(r + 1)       > 1.5
        c2 = (r / 50) ** 2     + np.cos(s / 7)            > 1.2
        c3 = np.sin(t / 10)    - (r / 100) ** 1.5         < 0.5
        c4 = (s / 14) * np.log(t)                         < 0.8
        return c1, c2, c3, c4

    # ------------------------------------------------------------------
    # 2 generate samples until every class is full
    # ------------------------------------------------------------------
    features, labels = [], []            # temporary storage
    counts = [0] * 5

    while any(cnt < tgt for cnt, tgt in zip(counts, per_c)):
        t   = rng.uniform(5, 35)
        r   = rng.uniform(0, 200)
        s   = rng.uniform(0, 14)
        c1, c2, c3, c4 = predicates(t, r, s)

        # exclusive assignment
        if   c1 and counts[0] < per_c[0]:
            label = 0
        elif c2 and counts[1] < per_c[1]:
            label = 1
        elif c3 and counts[2] < per_c[2]:
            label = 2
        elif c4 and counts[3] < per_c[3]:
            label = 3
        elif counts[4] < per_c[4]:       # fallback ‘other’
            label = 4
        else:
            continue                      # skip, class already full

        features.append((t, r, s))
        labels.append(label)
        counts[label] += 1

    X = np.asarray(features)
    labels = np.asarray(labels)

    # ------------------------------------------------------------------
    # 3 yield functions
    # ------------------------------------------------------------------
    y = np.zeros(n)
    idx = labels == 0
    y[idx] = 8 + 2 * np.sin(X[idx, 0] / 5)      + 0.01 * X[idx, 1]
    idx = labels == 1
    y[idx] = 5 + 0.02 * X[idx, 1]               + 1.5 * np.cos(X[idx, 2] / 3)
    idx = labels == 2
    y[idx] = 3 + 0.5 * (X[idx, 2] / 14) ** 2    - 0.01 * X[idx, 0]
    idx = labels == 3
    y[idx] = 4 + 0.02 * X[idx, 2]               + 0.1 * np.log(X[idx, 1] + 1)
    idx = labels == 4                           # simple baseline
    y[idx] = 4 + 0.01 * X[idx, 2]

    # ------------------------------------------------------------------
    # 4 add noise & build masks
    # ------------------------------------------------------------------
    y_noisy = y + rng.normal(0, (noise / 100) * y.std(ddof=0), size=n)

    one_hot = np.eye(5, dtype=bool)[labels]
    masks   = [one_hot[:, k] for k in range(5)]

    if verbose:
        print("Regime counts (Top/Mid/Dry/Cloud/Other):", counts)

    return X, y_noisy, masks, masks




# def load_synthetic12(n=600, seed=0, noise=0, verbose=False):
#     """
#     Synthetic dataset: Air quality index under nonlinear environmental factors.
    
#     Features:
#       - x[:,0]: Temperatur (°C, 0–40)
#       - x[:,1]: relative Luftfeuchtigkeit (0–1)
#       - x[:,2]: PM2.5-Konzentration (µg/m³, 0–200)

#     Regime-Bedingungen:
#       1. Sehr schlecht:     cos(pm/50) - humidity**2 < -0.5
#       2. Schlecht:          exp(-temp/20) + (pm/100)**1.5 > 1.2
#       3. Mäßig:             sin(humidity·π) + log(pm+1) < 0.8
#       4. Gut:               sonst
#     """
#     import numpy as np

#     np.random.seed(seed)
#     temp = np.random.uniform(0, 40, size=n)
#     hum  = np.random.uniform(0, 1, size=n)
#     pm   = np.random.uniform(0, 200, size=n)
#     x = np.column_stack([temp, hum, pm])

#     # Bedingungen
#     m1 = np.cos(pm/50) - hum**2 < -0.5
#     m2 = np.exp(-temp/20) + (pm/100)**1.5 > 1.2
#     m3 = np.sin(hum * np.pi) + np.log(pm+1) < 0.8
#     m4 = ~(m1 | m2 | m3)

#     # AQI-Funktionen
#     y = np.zeros(n)
#     y[m1] = 200 + 0.5*pm[m1] + 10*np.cos(temp[m1]/10)
#     y[m2] = 150 + 0.8*pm[m2] - 5*np.exp(-hum[m2])
#     y[m3] = 100 + 0.3*pm[m3] + 20*np.sin(temp[m3]/15)
#     y[m4] = 50 + 0.2*pm[m4] + 30*hum[m4]

#     # Rauschen
#     std = np.std(y)
#     y_noisy = y + np.random.normal(0, (noise/100)*std, size=n)

#     if verbose:
#         print("Regime-Counts:", m1.sum(), m2.sum(), m3.sum(), m4.sum())

#     masks = [m1, m2, m3, m4]
#     return x, y_noisy, masks, masks


# def load_synthetic10(n=600, seed=0, noise=0, verbose=False):
#     """
#     Synthetic stress-strain dataset for 3 materials: ABS plastic, Aluminum 6061-T6, Mild steel.
#     Balanced elastic/plastic per material → 6 masks.

#     - Features:
#         x[:,0]: strain ε (0 to 0.2)
#         x[:,1]: temperature T (°C, 20 to 500)
#         x[:,2]: strain rate ε̇ (s⁻¹, 0.001 to 10)
#         x[:,3]: material index (0=ABS,1=Al,2=Steel)
#     - Target:
#         stress σ (MPa)
#     Materials properties (at 20°C):
#       ABS: E≈2.35 GPa, σ_m≈44.8 MPa  
#       Al6061-T6: E≈68.9 GPa, σ_m≈276 MPa  
#       Mild steel: E≈210 GPa, σ_m≈350 MPa
#     Hardening coeff H₀, exponent n₀, and strain rate sensitivity m₀ chosen to reflect typical work-hardening.
#     """
#     np.random.seed(seed)
    
#     # 1) assign materials equally
#     mats = np.repeat(np.arange(3), n//3)
#     np.random.shuffle(mats)
    
#     # 2) sample temp and rate
#     temp = np.random.uniform(20, 500, n)
#     rate = np.random.uniform(0.001, 10, n)
    
#     # 3) base props per material
#     E0 = np.array([2.35e3, 68.9e3, 210e3])  # MPa
#     sigma_m = np.array([44.8, 276, 350])   # MPa
#     H0 = np.array([100, 1000, 1500])        # MPa
#     n0 = np.array([0.1, 0.2, 0.3])
#     m0 = np.array([0.02, 0.03, 0.04])       # strain rate sensitivity exponents
    
#     # map to samples
#     E0_s = E0[mats]
#     sigma_m_s = sigma_m[mats]
#     H0_s = H0[mats]
#     n0_s = n0[mats]
#     m0_s = m0[mats]
    
#     # temp dependence (linear degrade)
#     E = E0_s * (1 - 0.0003 * (temp - 20))
#     sigma_yield = sigma_m_s * (1 - 0.0002 * (temp - 20))
#     epsilon_yield = sigma_yield / E
#     H = H0_s * (1 - 0.0005 * (temp - 20))
#     n_exp = n0_s + 0.0001 * (temp - 20)
    
#     # 4) sample strain for balanced regimes per material
#     strain = np.empty(n)
#     elastic = np.random.rand(n) < 0.5
#     # below yield
#     strain[elastic] = np.random.uniform(0, 0.9 * epsilon_yield[elastic])
#     # above yield
#     strain[~elastic] = np.random.uniform(1.1 * epsilon_yield[~elastic], 0.2)
    
#     # assemble features
#     x = np.column_stack([strain, temp, rate, mats])
    
#     # 5) compute stress
#     stress = np.empty(n)
#     # Elastic regime
#     stress[elastic] = E[elastic] * strain[elastic]
#     # Plastic regime with strain rate effect
#     delta = strain[~elastic] - epsilon_yield[~elastic]
#     strain_rate_ref = 0.001
#     strain_rate_factor = 1 + m0_s[~elastic] * np.log(rate[~elastic] / strain_rate_ref)
#     strain_rate_factor = np.maximum(strain_rate_factor, 1)  # ensure factor >= 1
#     stress[~elastic] = (
#         sigma_yield[~elastic]
#         + H[~elastic] * (delta ** n_exp[~elastic]) * strain_rate_factor
#     )
    
#     # 6) add noise
#     std = np.std(stress)
#     y_noisy = stress + np.random.normal(0, (noise/100)*std, size=n)
    
#     # 7) create 6 masks: for each material × regime
#     masks = []
#     for m in range(3):
#         masks.append((mats == m) & elastic)
#         masks.append((mats == m) & ~elastic)
    
#     print("Mask counts (ABS-elastic, ABS-plastic, Al-elastic, Al-plastic, Steel-elastic, Steel-plastic):") if verbose else None 
#     print([mask.sum() for mask in masks]) if verbose else None 
    
#     return x, y_noisy, masks, masks

# def load_synthetic12(n=600, seed=0, noise=0, balance=False, verbose=False):
#     """
#     Synthetic dataset mimicking one simplified SHA-256 step: a 4-input modular addition,
#     with optional balancing of overflow_count classes.

#     - Inputs (X):
#         X[:,0]: word w0 (unsigned 32-bit integer)
#         X[:,1]: word w1 (unsigned 32-bit integer)
#         X[:,2]: word w2 (unsigned 32-bit integer)
#         X[:,3]: word w3 (unsigned 32-bit integer)
#         X[:,4]: overflow_count (number of times the 32-bit sum wrapped modulo 2^32)
#     - Target (y):
#         sum_mod = (w0 + w1 + w2 + w3) mod 2^32 (decimal), with optional noise

#     - balance: if True, samples so that each overflow_count class (0,1,2,3,...) is approximately equally represented.
#     - verbose: if True, prints overflow count distribution
#     """
#     rng = np.random.default_rng(seed)
#     mod_base = 2**32

#     def sample_batch(k):
#         # sample k raw words, return X_words, sum_raw
#         Xw = rng.integers(0, mod_base, size=(k, 4), dtype=np.uint64)
#         sum_r = Xw.sum(axis=1)
#         return Xw, sum_r

#     if not balance:
#         # uniform sampling
#         X_words, sum_raw = sample_batch(n)
#     else:
#         target_classes = 4  # wraps 0,1,2,3
#         per_class = int(np.ceil(n / target_classes))
#         X_list, y_list = [], []
#         counts = {i: 0 for i in range(target_classes)}
#         # keep sampling until each class has per_class samples or pool large enough
#         while sum(counts.values()) < per_class * target_classes:
#             Xw, sum_r = sample_batch(per_class * target_classes)
#             wraps = (sum_r // mod_base).astype(int)
#             for i in range(target_classes):
#                 mask = wraps == i
#                 needed = per_class - counts[i]
#                 if needed > 0:
#                     sel = np.where(mask)[0][:needed]
#                     for idx in sel:
#                         X_list.append(Xw[idx])
#                         y_list.append(sum_r[idx])
#                         counts[i] += 1
#             # break if enough
#         # combine and truncate to n
#         X_words = np.array(X_list[:n], dtype=np.uint64)
#         sum_raw = np.array(y_list[:n], dtype=np.uint64)

#     # compute overflow_count and modulo sum
#     overflow_count = (sum_raw // mod_base).astype(np.uint32)
#     sum_mod = (sum_raw % mod_base).astype(np.uint64)

#     # Divide sum_mod by 1e9 and the variables as well, so that we can see normal values 
#     sum_mod = sum_mod / 1e9
#     X_words = X_words / 1e9
    
#     # add noise if requested
#     if noise > 0:
#         y = sum_mod.astype(float) + rng.normal(0, noise, size=n)
#         y = np.floor(y).astype(np.uint64) % mod_base
#     else:
#         y = sum_mod

#     # assemble X including overflow count
#     X = np.column_stack([X_words, overflow_count])

#     if verbose:
#         unique, counts = np.unique(overflow_count, return_counts=True)
#         print("Overflow count distribution:")
#         for u, c in zip(unique, counts):
#             print(f" wraps={u}: {c} samples")

#     # masks 
#     masks = [overflow_count == i for i in np.unique(overflow_count)]

#     return X, y, masks, masks 