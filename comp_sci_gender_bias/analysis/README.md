## Sentence clusters

To generate boxplots showing the proportion of tokens in each sentence category per subject, run:

```bash
python comp_sci_gender_bias/analysis/sentence_clusters.py
```

This will create individual subject boxplots and the underlying data in `outputs/figures/sentence_clusters/`.

# Analysis

## Create and save charts and the related data

To create and save charts and the related data for:

- Girls entry percentage into GCSE subjects
- Mean gender differences across each POS
- School level mean gender differences
- Scatterplots of secondary data with mean gender difference

Run:

```bash
python comp_sci_gender_bias/analysis/save_charts_data.py
```
