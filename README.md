File descriptions

1. __init__.py
Initializes the denoising module directory so components can be imported as a package.

3. cnn_denoiser.py
Defines a 1D CNN-based denoiser (currently scaffolded/identity) intended for future learning-based signal denoising.

4. denoise_evaluation_summary.json
Stores aggregated evaluation results and summary statistics from denoising performance tests.

5. denoise_folder.py
Applies the denoising pipeline batch-wise to a folder of chromatogram files and writes denoised outputs and QC metadata.

6. denoise_router.py
Central controller that selects between classical and CNN denoisers, runs QC checks, and enforces safe fallback logic.

7. evaluate_denoising.py
Provides utility functions to load chromatogram data and compute quantitative metrics for denoising evaluation.

9. image.png
Illustrative diagram showing the overall denoising pipeline architecture and data flow.

10. mocca_denoise.py
Implements classical, deterministic denoising methods used as the production-safe baseline.

11. qc_metrics.py
Computes quality-control metrics such as peak area distortion and baseline noise changes to validate denoising safety.

12. run_evaluation_router.py
Runs end-to-end evaluations of the full denoising router (classical vs CNN) and records QC and noise-reduction statistics.

13. run_evaluation.py
Executes standalone evaluation of classical denoising methods over multiple samples and summarizes the results.

14. compare_methods
Compare the CNN and the classical denoising

Input folder has sample input files from ADE, output has sample files that will go to MOCCA
