# IMapbook: Automating Analysis of Group Discussions

The constant advancements in technology introduce new approaches and tools for students' education.
 Such tool is the ImapBook, an interactive platform that allows students to communicate and discuss about 
 the book they are reading. To give the best support, such applications need natural language processing 
 features, that understand the content of communications and automatically intervene when the focus on the 
 topic is lost. Thus, we present different text classification models that classify such texts from students' 
 communications into useful categories. The best performing method is fine-tuned BERT, which outperforms simpler 
 classification methods based on handcrafted features.

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
virtualenv venv
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

Download our pre-trained BERT models from [here](https://unilj-my.sharepoint.com/:u:/g/personal/pk0404_student_uni-lj_si/EeuW9GdbAxRAt0PuM80eAnwByKcI-ccNHpBAO_W5H2jW9w?e=oBbwTD)
 and store them in the folder 
`/src/classifier_BERT/pretrained_models/`.

### Running

Go to src folder.

```
cd src
```

There are two groups of models, baseline models and deep neural models. To evaluate baseline models, run the following
command. Script stores numerical results to `./results/results_baselines.yaml` and plot to `./results/plot_baselines.pdf`. 

```
python3 evaluate_baselines.py
``` 

To evaluate ELMo and BERT models, run the following command. Results are stored in 
file `./results/results_deep_models` and plotted
in `./results/plot_deep_models.pdf`.
 

```
python3 evaluate_deep_models.py
```

To plot features importance, run next command. Figure is saved to `./results/plot_imp_RF_plot.pdf`. 

```
python3 plots/feature_importances.py
```

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
