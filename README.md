# Gender Bias in GCSE Options

<img
  src="https://github.com/nestauk/comp_sci_gender_bias/blob/dev/outputs/figures/girls_entry_percentage/girls_entry_percentage_img.png?raw=true"
  width="400"
  style="display: block;
         margin-left: auto;
         margin-right: auto;"
/>

There is a disparity in the proportion of students choosing to take different subjects at GCSE level according to their gender. Typically, boys are overrepresented in subjects like computer science, while the same is true for girls in subjects like drama. Some subjects, such as geography exhibit a more even split in uptake between boys and girls.

In this project, we examine the text from GCSE subject descriptions, shown to students at the time they are choosing which courses to pursue, as a potential driver or indicator of biasses that lead to disparities in uptake. The descriptions are taken from a number of schools and are used to inform students about the GCSE options that they have to choose from. While the messages in these texts may not be the deciding factor for students' choices, they may be indicative of portrayals of the subjects within a school and wider society. Analysing whether differences exist between subjects is one way to inform interventions that are within schools' immediate control.

## Project outline

The project is formed of several components, including data collection, processing and analysis:

- **Data collection**: A number of course descriptions were obtained by The Behavioural Insights Team (BIT) for computer science and geography courses from around 180 schools. This was supplemented by data scraped from approximately 50 additional school websites by Nesta, covering computer science, geography and drama.
- **Gender biassed terms**: BIT carried out an initial analysis to explore the language used in different GCSE option texts. The work here is a close reproduction of that work. We do a pairwise subject comparison in which the most frequent words in a subject relative to the other subject are calculated. A gender difference measure for each word is calculated by measuring the average cosine distance to a set of gendered words ('he', 'she', etc.) using pretrained GloVe embeddings to reveal the relative difference in the prevalence of gendered terms in each subject.
- **Gender biassed descriptions**: We repeat the method above, and take the average male-female distance score for all terms in each description. This gives a school level measure of the degree of gender biassed langauge in each subject.
- **Semantic similarity of descriptions**: By creating vector representations of entire course descriptions, we aimed to explore linguistic similarities and differences in the ways that schools described the subjects on offer.
- **Sentence clustering**: Preliminary exploration of clustering sentence level embeddings to analyse the ways that courses are described in a subject agnostic way. Sentence embeddings are clustered based on semantic similarity and the clusters are grouped into categories (e.g. "course content", "motivating factors") to capture the messages being given within a description with a higher level of nuance.
- **Readability**: As an additional point of interest, we explored the readability of each course description using the Flesch Reading Ease and Dale-Chall Readability scores.

## Navigating the code and results

The work for this project is split into two main components:

- Pipeline - the data collection and processing components of the project.
- Analysis - produces charts and summary statistics

All of the final charts and tables can be found in `outputs/`.

There is also a [slide deck with a detailed discussion of the results](https://docs.google.com/presentation/d/1mhpjyglXV-naLgwFBQ5z2CF1g7zgcWSEuNcjec_eCdM/edit#slide=id.g13196716ea3_0_0) (Nesta and BIT access only).

## Setup

Want to reproduce these results, develop them further, or add new analysis and features? To set up the project repository on your machine, follow the steps below:

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Clone this repository
- Create a blank cookiecutter conda log file:
  - `mkdir .cookiecutter/state`
  - `touch .cookiecutter/state/conda-create.log`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

## Questions and feedback

If you want to get in touch, [send us a message](mailto:data_analytics@nesta.org.uk).

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
