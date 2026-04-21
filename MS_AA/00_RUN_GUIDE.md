# MS-AA Notebook Suite

Use the same two inputs everywhere:

- `ANALYSIS_TYPE = "spatial"` or `"temporal"`
- `FIT_SCOPE = "within"` or `"across"`

## Run order

1. `01_fit_msaa_flexible.ipynb`
2. `02_decode_msaa_flexible.ipynb`
3. Matching post-processing notebook:

- `03_postprocess_across_spatial.ipynb`
- `04_postprocess_across_temporal.ipynb`
- `05_postprocess_within_spatial.ipynb`
- `06_postprocess_within_temporal.ipynb`

## Logic

- Shared notebooks:
  - fitting
  - decoding

- Branching notebooks:
  - across-condition post-processing keeps clustering where it makes sense
  - within-condition post-processing drops cross-condition clustering
  - spatial AA emphasizes temporal motifs + spatial coefficients
  - temporal AA emphasizes spatial motifs + time-varying coefficients