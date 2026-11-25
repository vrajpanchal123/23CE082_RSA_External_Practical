# Privacy-Preserving Digit Recogniser (DP-SGD, Opacus)

Responsible and Safe AI (AIML4001) - External Practical, Set D
Name: Vraj Panchal | Enrollment/Roll No: 23CE082 | Batch: D1

Trains a GroupNorm CNN on MNIST (20,000-image subset): non-private baseline, DP models
for eps = 10, 5, 1 (delta=1e-5, clip=1.0, 3 epochs), a fixed noise run (sigma=1.1) with
eps per epoch, and a membership-inference confidence check.

## Run
    pip install torch torchvision opacus matplotlib numpy
    python dp_digit_recogniser.py        # MNIST_DIR (default ./data), OUT_DIR (default .)

## Files
- dp_digit_recogniser.py - full program
- accuracy_vs_epsilon.png, epsilon_vs_epoch.png - plots
- results.json, run_log.txt - recorded results
