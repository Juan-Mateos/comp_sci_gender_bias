# Pipeline

## Process school description documents

BIT school descriptions for computer science and geography are stored as `docx` files in `inputs/data/`.

In order to parse them, run:

```bash
python comp_sci_gender_bias/pipeline/text_data/parse_bit_data.py
```

This creates two `json` lookups between school ids and school descriptions. Note that the school descriptions have not been processed in any way beyond stripping whitespaces at the beginning or end.

You can get these id_lookups, as well as a lookup table between school ids, names and other metadata, using `text_descriptions` and `school_table` from `comp_sci_gender_bias/getters/school_data.py`.
