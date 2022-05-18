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

## Make sentence embeddings of school course descriptions

To make and save the school course description sentence embeddings, run:

```bash
python comp_sci_gender_bias/pipeline/sentence_embeddings/create_sentence_embeddings.py
```

This will save the embeddings to `comp_sci_gender_bias/outputs/embeddings`.

These embeddings can be loaded using the `load_embedding` getter, for example:

```python
from comp_sci_gender_bias.getters.embedding import load_embedding

geo_embedding = load_embedding(subject="geo")
cs_embedding = load_embedding(subject="compsci")
```
