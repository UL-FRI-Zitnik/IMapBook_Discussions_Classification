# IMapbook: Automating Analysis of Group Discussions

Nature Language Processing has seen a huge rise in popularity in 
recent years. It is now broadly studied topic with many successful 
applications. In this project we touch subfield Text Classification 
and apply its methods to the data from IMapbook, a web-based
technology that allows reading material to be intermingled with interactive 
games and discussions. Some portion of discussions from this platform were manually 
annotated, each reply was given more categories based on the information in the 
reply. Our goal is to  take this data and try to build a classifier which would 
predict these categories. Such classifier could then be used to automate analysis 
of discussions at this platform, recommend the time for the teacher’s intervention
and more.

### Data

The dataset is provided by IMapBook and includes the discussions between students and teachers
on the topics of the book they are reading. The dataset includes approximately 3500 Slovene messages, 
from 9 different schools and on 7 different books, which were also translated to English. 
Students in each school were divided in "book clubs", where the conversations occurred.
The data was manually annotated, with three main tags:

*  _Book Relevance_: Whether the content of the message is relevant to the topic of the book discussion.
* _Type_: Whether the message is a question (Q), answer (A), a mix of the two (AQ, QA) or a statement (S).
* _Category_: Whether the message is a simple chat message (C), related to the book discussion (D), 
moderating the discussion (M), wondering about users' identities (I), referring to a task, switching 
it or referring to a particular position in the application (S), or other cases (O).

### Installation

##### Python Dependencies

We suggest running everything in virtual environment. If you don't have `virtualenv` installed, run

```
pip3 install virtualenv
```

To create and activate virtual environment, run

```
python3 -m venv venv
source venv/bin/activate
```

Then, to install all Python dependencies, run

```pip3 install -r requirements.txt```

If you just want to evaluate models or if you don't have NVIDIA GPU, that's it. If you would like
 to train BERT models, install CUDA by

```
pip3 install cudatoolkit~=10.1
```

##### Downloading ELMo and BERT pre-trained Models

To download ELMo model trained on Slovene corpus, visit [clarin.si](http://hdl.handle.net/11356/1277)
website and download *Slovenian ELMo model* (file `slovenian-elmo.tar.gz`). Extract its contents and place 
files `options.json` and `slovenian-elmo-weights.hdf5` into `data/elmo/` folder.

Download our pre-trained BERT models from [here]() and store them in the folder 
`/src/classifier_BERT/pretrained_models/`.

### Running

Go to src folder.

```
cd src
```

There are two groups of models, baseline models and deep neural models. To run baseline models, run

```
python3 evaluate_baselines.py
``` 

Script stores numerical results to `./results/results_baselines.yaml` and plot to `./results/plot_baselines.pdf`.

By

```
python3 evaluate_deep_models.py
```

ELMo and BERT models are evaluated. Results are stored in file `./results/results_deep_models` and plotted
in `./results/plot_deep_models.pdf`.


To plot features importance, run 

```
python3 plots/feature_importances.py
```

Figure is saved to `./results/plot_imp_RF_plot.pdf`.

To train BERT model, run 

```
python3 train_bert.py
```

### Repository Structure

* `data/`: placeholder for IMapBook data, ELMo embeddings, lexicon, cached data, 
Slovene stop words, preprocessed data, Slovenian names.
* `src/`: Source code. Models are stored in folders, prefixed with `classifier_`.
* `results`: Numerical and graphical results of our research.
* `report/report.pdf`: Report. 
* `relevant articles/`: Articles relevant to our work.
