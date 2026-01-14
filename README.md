File descriptions

1. __init__.py
Initializes the denoising module directory so components can be imported as a package.

2. cnn_denoiser.py
Defines a 1D CNN-based denoiser (currently scaffolded/identity) intended for future learning-based signal denoising.

3. denoise_evaluation_summary.json
Stores aggregated evaluation results and summary statistics from denoising performance tests.

4. denoise_folder.py
Applies the denoising pipeline batch-wise to a folder of chromatogram files and writes denoised outputs and QC metadata.

5. denoise_router.py
Central controller that selects between classical and CNN denoisers, runs QC checks, and enforces safe fallback logic.

6. evaluate_denoising.py
Provides utility functions to load chromatogram data and compute quantitative metrics for denoising evaluation.

7. image.png
Illustrative diagram showing the overall denoising pipeline architecture and data flow.

8. mocca_denoise.py
Implements classical, deterministic denoising methods used as the production-safe baseline.

9. qc_metrics.py
Computes quality-control metrics such as peak area distortion and baseline noise changes to validate denoising safety.

10. run_evaluation_router.py
Runs end-to-end evaluations of the full denoising router (classical vs CNN) and records QC and noise-reduction statistics.

11. run_evaluation.py
Executes standalone evaluation of classical denoising methods over multiple samples and summarizes the results.
