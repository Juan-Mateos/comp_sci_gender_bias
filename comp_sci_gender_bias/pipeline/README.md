# Pipeline

## Process school description documents

BIT school descriptions for computer science and geography are stored as `docx` files in `inputs/data/`.

In order to parse them, run:

```bash
python comp_sci_gender_bias/pipeline/text_data/parse_bit_data.py
```

This creates two `json` lookups between school ids and school descriptions. Note that the school descriptions have not been processed in any way beyond stripping whitespaces at the beginning or end.

You can get these id_lookups, as well as a lookup table between school ids, names and other metadata, using `text_descriptions` and `school_table` from `comp_sci_gender_bias/getters/school_data.py`.

## Process text

Download the GloVe embeddings to a special folder (will take some time and downloads a 870 MB file).

```bash
export GLOVE_PATH=~/glove-embeddings
mkdir $GLOVE_PATH
python -m wget "https://nlp.stanford.edu/data/glove.6B.zip" -o $GLOVE_PATH
unzip $GLOVE_PATH/glove.6B.zip -d $GLOVE_PATH && rm $GLOVE_PATH/glove.6B.zip
```

## Calculate girls entry percentage into GCSE subjects

To calculate the girls entry percentage into GCSE subjects, run:

```bash
python comp_sci_gender_bias/pipeline/subject_entry/girls_subject_entry.py
```

This will save a csv to `comp_sci_gender_bias/outputs/girls_entry_percentage/girls_entry_percentage.csv`

## Make male - female differences for most frequent subject words

To produce csv files comparing two subjects with columns for:

- Top most frequent words in one subject relative to the other subject
- Part of Speech tag
- Word frequency across both subjects' course descriptions
- Word count across both subjects' course descriptions
- Male - female difference (calculated as the average cosine similarity to masculine words - average cosine similarity to feminine words)

Run:

```bash
python comp_sci_gender_bias/pipeline/glove_differences/make_differences.py
```

Results are produced for Computer Science and Geography (using data collected by BIT) and Computer Science and Drama (using data collected by Nesta) for the top adjectives/adverbs, nouns and verbs. The results are saved to csv files in `outputs/tables/most_frequent_word_gender_differences/`.

## Make male - female mean differences for each POS and corpus

To produce the mean male - female mean differences for each POS and corpus (for both BIT and Nesta collected data), run:

```bash
python comp_sci_gender_bias/pipeline/glove_differences/make_mean_differences.py
```

The results are saved to csv files in `comp_sci_gender_bias/outputs/mean_differences/`.

The results can be loaded with getter at `comp_sci_gender_bias.getters.mean_gender_differences.mean_gender_differences`

## Make sentence embeddings of school course descriptions

To make and save the school course description sentence embeddings, run:

```bash
python comp_sci_gender_bias/pipeline/sentence_embeddings/create_sentence_embeddings.py
```

This will save the embeddings to `comp_sci_gender_bias/outputs/embeddings/`.

These embeddings can be loaded using the `load_embedding` getter, for example:

```python
from comp_sci_gender_bias.getters.embedding import load_embedding

geo_embedding = load_embedding(subject="geo")
cs_embedding = load_embedding(subject="compsci")
```

## Combine Department for Education datasets

Department for Education datasets are stored in `inputs/data/dfe_school_info`. They include information about schools including: unique school reference number, location, ofsted rating, gender split, average attainment by gender.

To combine these datasets together, run:

```bash
python comp_sci_gender_bias/pipeline/additional_school_info/combine_dfe_school_data.py
```

This will save a file `dfe_combined_dataset.csv` in `inputs/data/dfe_school_info/`

This file can be loaded using the getter `comp_sci_gender_bias.getters.dfe_combined_school_data.dfe_combined_school_data`

## Create lookup for school name and school unique reference number

The `school_master_table.csv` file in `inputs/data/` does not have a column for school unique reference number and so cannot be easily joined to datasets which use the school unique reference number.

To create a lookup containing `school_name` and `school_unique_reference_number`, run:

```bash
python comp_sci_gender_bias/pipeline/pipeline/urn_to_school_name_lookup/urn_to_school_name_lookup.py
```

This will save a file `urn_school_lookup_full.csv` to `inputs/data/urn_school_lookups/`

This file can be loaded using the getter `comp_sci_gender_bias.getters.urn_school_lookup.urn_to_school_name_lookup`

## Make dataset containing scraped subject descriptions, mean gender difference and secondary DfE data

To make a dataset containing the scraped subject descriptions, school level mean gender difference for each subject and the secondary DfE data, run:

```bash
python comp_sci_gender_bias/pipeline/school_level_gender_bias/make_school_lvl_gender_bias.py
```

This will save a csv file `scraped_schools_urn_dfe.csv` to `comp_sci_gender_bias/outputs/school_level/`.

This file can be loaded using the getter `comp_sci_gender_bias.comp_sci_gender_bias.getters.school_lvl_bias_with_dfe_data.school_lvl_bias_with_dfe_data`

## Calculate readability of course descriptions scraped by Nesta and export charts and tables

To calculate the readability scores and generate the chart and table outputs, run:

```bash
python comp_sci_gender_bias/pipeline/readability/subj_readability.py
```

This will generate boxplots and their underlying data in `outputs/figures/readability` and tables of descriptive statistics and description examples in `outputs/tables/readability`.

## Generate sentence clusters for manual grouping

:warning: Running this will overwrite previous results, which may contain manually grouped clusters. It is not advised to run this step unless you plan to revise the cluster categories.

If this step is run, the process for grouping the clusters can be found and used in the notebook, `comp_sci_gender_bias/analysis/sentence_cluster_inspection.ipynb`.

To split the course descriptions (scraped by Nesta) into sentences, vectorise the sentences and cluster them, run:

```bash
python comp_sci_gender_bias/pipeline/sentence_clusters/create_clusters.py
```

This will save three csv files `{subj}_sentence_clusters.csv` to `comp_sci_gender_bis/outputs/sentence_clusters`.

After running this, the clusters will need to be manually inspected and grouped into categories.
