# Collecting Additional Course Descriptions

## Overview

Within this directory, there is a dataset of 50 schools' course descriptions for Geogrpahy, Computer Science, and additionally, Drama, taken from school websites using the below approach.

## Methods

```
import requests

relevant_pages = []
for site in websites:
    try:
        url = site + "/year_9_options/"
        response = requests.get(url, allow_redirects=False)
        if response.status_code == 200:
            relevant_pages.append(url)
    except:
        pass
    try:
        url = site + "/year_8_options/"
        response = requests.get(url, allow_redirects=False)
        if response.status_code == 200:
            relevant_pages.append(url)
    except:
        pass
```

The code is very simple, but can be slow. To combat this, chunks of sites were executed in parallel using Metaflow.

The output of this was to produce a list of pages that programtically looked like they both held Options information, and were scrapable.

## Challenges

The main challenge by far was entropy of school websites, and locations of options information within. Lots of valid pages were returned by this method, but amongst them were lots of useless websites (the `allow_redirects=False` was one way of reducing these, but on websites that had been shut down this didn't work). Furthermore, acquiring relevant data within the page proved challenging too, due to there being no consistent place where useful course description data was held. Because of this, the approach became semi manual from here, producing the results. With this method, 1 course description per minute was ascertained.

## Other approaches

There may actually be a simpler approach than described above, discovered too late to explore, but search engine results of "Year 9 (or Year 8) Options Booklets" returned almost entirely relevant results; PDF links of options booklets. Leveraging a Search Engine API and collecting results would produce a good set of PDFs to parse for additional data (although still semi manual).
