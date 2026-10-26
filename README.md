# H9T application benchmark — preliminary v0.3

Run: `python benchmark.py`. Dependencies: numpy, scipy, scikit-image, scikit-learn, and the H9T source file at the path imported by the script. Change the absolute source path before moving to another machine.

## Protocol
- H9T: normalized nine-point Helmert basis.
- DCT-II: scipy.fft.dct, type=2, norm='ortho'.
- Identity: no basis change, as a control.
- Image: 243×243 top-left crops of three scikit-image samples, non-overlapping 9×9 blocks, global top-|coefficient| selection at 5%, 10%, and 20%, PSNR. No entropy coding, quantization, perceptual rating, adaptive block selection, or boundary correction. Therefore, not a codec comparison.
- Signal: synthetic fixed-seed periodic and burst signals, additive Gaussian noise sigma=0.08, non-overlapping length-9 blocks, global largest coefficient selection, MSE against synthetic clean signal. Rate selection was not tuned; two examples do not give robust statistical confidence.
- Anomaly: injected impulses and block-level shifts in noisy periodic data. Non-DC coefficient-energy score, ROC-AUC. Because the two orthonormal bases share their constant vector, that score must be the same to floating-point tolerance (Parseval); this is a negative control, not independent performance confirmation.
- Timing in CSV is not a balanced performance comparison and should not be used for speed claims.

## Findings
1. DCT-II outperformed this baseline H9T at 10% retained coefficients on all three images (roughly 3.2 dB PSNR).
2. H9T and DCT-II delivered nearly identical MSE in two toy signal denoising cases at 40% retention.
3. Spikes were perfectly separated in the synthetic anomaly experiment, but this did not distinguish the algorithms. Level shifts were near random.

## Important limitations
- Only three image crops and two synthetic signal models; no confidence intervals, holdout data, train/validation/test split, learned models, real anomaly datasets, or robust timing runs.
- Wavelet baselines (Haar/Daubechies via PyWavelets) not run because the package was unavailable; must be added before claims against wavelets.
- No ML classifier or AI benchmark was executed. Anomaly scoring is not an AI model.
- H9T's application advantage has NOT been established. An improved transform or domain-specific algorithm should be preregistered and evaluated on held-out data against DCT, wavelets, FFT, and no-transform baselines.
