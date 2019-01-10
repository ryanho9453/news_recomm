# Chinese news recommendation

The model recommend relevant chinese news according to your keyword


# Overview

  - Input some keyword in config and the model will recommend some relevant news
  - you can set multiple keyword sets to follow multiple topics
  - We train the word embedding by using LDA and RNN



# model
### word embedding
Use the word embedding in this [project](https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=) to vectorize each news
> [Chinese keyword recommendation for search engine](https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=)

the word embedding is trained with LDA and RNN, the parameter could be tune in config.json



# Demo

Download [data](https://drive.google.com/file/d/1vFqXSLDlglzIJ3ilBLeuR8iCf7gtPSZk/view?usp=sharing) and put in folder "data"

Download [pre-trained RNN](https://drive.google.com/file/d/1wqj1-ZAxq12TpyqRs6ZKIvTdOFkZ6FgK/view?usp=sharing) and put in folder "preprocess/word_vector/models/RNN/saved_model/"

then

#### 1. Ask for a recommendation with saved model

Input your words of a topic and give it a name.
Example : 

<a href="https://ibb.co/6YqgCQ1"><img src="https://i.ibb.co/F6dgTp0/2019-01-03-12-13-45.png" alt="2019-01-03-12-13-45" border="0"></a>

and then
               
```sh
$ python3 main.py --topic-name (input your topic name here)
```


#### 2. Ask for a recommendation after changing the topic
add "--reset" after you change the words in your topic

```sh
$ python3 main.py --topic-name (input your topic name here) --reset
```





