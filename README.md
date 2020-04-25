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

The _Category_ category can be further on split in sub-categories; _chats_ may be in the form of greetings (G), 
related to the book (B), they could be encouraging (E), talk about feelings (F), contain cursing (C) or others (O), 
_Discussion_ messages could be questions (Q), answers (A), answers to users, still related to the discussion topic (AA) 
or encouraging the discussion (E); _identity_ messages can be answers(A), questions (Q) or their combination (QA).
The dataset is suitable for both binary and multi-class classification, whether the target variable is the relevance
 or the category of the message respectively.

### Installation

Run `pip3 install -r requirements.txt`.

### Running

Run by

```
cd src
python3 main.py
``` 

### Repository Structure

* `data/` contains the dataset and information about it. 
* `relevant articles/` contains articles relevant to our work.
* `src/` contains all source code to replicate our work.
* `report.pdf` 
