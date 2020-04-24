# Differentiating Between Two Opposing Reddits

By: Christopher Kuzemka: [Github Repository](https://git.generalassemb.ly/chriskuz/project_3)

## Problem Statement

My girlfirned loves to use Reddit. One of her favorite subreddits is ["r/aww"](https://www.reddit.com/r/aww/), a community dedicated forum largely consisting of cute animals and cute moments captured on video and on camera. However, there is another reddit that is the complete opposite of cute animals and cute moments captured on video and on camera -- this subreddit is known as ["r/natureismetal"](https://www.reddit.com/r/natureismetal) -- and it was merged together with "r/aww" to create a "super-subreddit" known as "r/dangerouslycute." The official reason for doing so is unknown, but the top rumor for the merger narrates that both moderators from each subreddit felt that they had enough of a mutual following to justify the merge and consolidate all posts together. This made my girlfriend very upset, as she was never a fan of the content from "r/natureismetal" and now she is tainted by its controversial content. We can also imagine that many others must also feel the same way about the merge. 

As a good data scientist who wishes to make his significant other happy, I have decided to help her make an app that will run Javascript in the background with Reddit and ultimately separate the consolidated subreddit content. She will be writing all other code necessary to create the app, while we will explore the jumbled data of "r/dangerouslycute" and help create the model that will separate the two subreddits from this super-subreddit.

Using data collected previously from the subreddits before the merge, we are going to utilize Natural Language Processing classification models to separate the subreddit content. Our supervised learning models will be judged by their accuracy measure for success. The models we will explore will be LogisticRegression, Multinomial Naive Bayes, and Gaussian Naive Bayes with a use of CountVectorizer and TFIDFVectorizer across our titular data. We will do an in depth analysis on a successful model and explore the various quirks behind the influences of its predictions.

## Executive Summary
A study was conducted where we analyzed the problem of a merged subreddit known as r/dangerouslycute. r/dangerouslycute is a ficional subreddit which includes content from the intimidating and graphic subreddit known as r/natureismetal and and includes content from the happy and positive-intent subreddit known as r/aww. In a sense, each of these subreddits' collective images are polar opposites of one another. They particularly do not share the same Rules and Guidelines and have a stark difference in types of content typically displayed. However, we are unsure of how this merger came to fruition and are now tasked with the problem of generating a model to filter the subreddit. We hope to create a successful model that can be incorporated into a new application which will run in the background of Reddit and eventually wish to generalize such app to to work with other subreddits. 

To gather the data, we utilized the pushshift Reddit API to collect submission data from the two non-fictional subreddits mentioned above. Data such as comment numbers per post, submission scores, classification metrics for appropriate content, date creation, author data, submission titles, and much more were collected for analysis. Much of the data was non-null value, but null text data were imputed with "spaces" for columns where necessary. Much attention was given towards the popularity of the submissions we collected to help understand any biases that would exist in the data. We even discovered which words were most frequent across each subreddit. 

Finally, we constructed three general models (not including a baseline) and tested different vectorizing methods across each model. We also tested a variety of parameters across our models and conducted cross validated gridsearchs to ultimately discover which model performed best with certain features. Ultimately, we were able to come up with a model to passively satisfy any requirements we would need to make a generalized subreddit filtering application, but also opened the idea for future considerations on how such model could be optimized.

## Table of Contents
[1.00 Data Loading](#1.00-Data-Loading)

[2.00 Data Cleaning and Analysis](#2.00-Data-Cleaning-and-Moderate-Analysis)

- [2.01 Quick Check](#2.01-Quick-Check)

- [2.02 Data Documentation Exploration](#2.02-Data-Documentation-Exploration)

- [2.03 Cleaning](#2.03-Cleaning)

- [2.04 Exploratory Data Analysis and Visualization](#2.04-Exploratory-Data-Analysis-and-Visualization)

[3.00 Machine Learning Modeling and Visulalization](#3.00-Machine-Learning-Modeling-and-Visulalization)

- [3.01 Model Preparation](#3.01-Model-Preparation)

- [3.02 Modeling](#3.02-Modeling)

- [3.03 Model Selection](#3.03-Model-Selection)

- [3.04 Model Evaluation](#3.04-Model-Evaluation)

[4.00 Conclusions](#4.00-Conclusions)

[5.00 Sources and References](#5.00-Sources-and-References)

|__Data Variable__|__Type__|__Significance__|
|---|---|---|
|`title`|__String Object__|*Submission title*|
|`subreddit`|__String Object__|*Native subreddit of submission*|
|`reddit_creation_identifier`|__Integer__|*ID for when submission was created*|
|`author`|__String Object__|*Submission author*|
|`number_of_comments`|__Integer__|*Number of comments on submission*|
|`score`|__Integer__|*Submission score*|
|`text_in_post`|__Boolean__|*Identifies if text is in post content*|
|`nsfw`|__Boolean__|*Identifies if submission is for mature audiences*|
|`author_flair_text`|__String Object__|*Subtitle for author*|
|`total_awards_received`|__Integer__|*Number of virtual awards a submission receives*|
|`timestamp`|__Datetime Object__|*Date of when submission was created*|

## Conclusions
The question comes down to whether we were able to create a model strong enough for app development. Before answering this question, let's recap on some of the things mentioned that would affect our data and modeling:

- our data sample was only 5000 submissions from subreddits with millions of users and thousands online at a given time (50/50 split for r/aww and r/natureismetal content)
- our data did not appear to be very popular content due to the lack of comments, high scores, and virtual awards (this means there was less scrutiny)
- our models were only based off the submission titles alone (lacking other helpful classification features, numeric weighting, text data, and image data)
- we did not explore every possible model known to existance and did not explore every parameter to help a model's accuracy

With a testing accuracy score of ~84% with our successful Multinomial Naieve Bayes model (incorporating a TFIDF Vectorizer), we are left with ~16% error in our model. In the world of large data, this sort of error will not be tolerated on the app market and our app will not be marketed as a successful app as a fair portion of posts we wish to differentiate on r/dangerouslycute will fail to be filtered. My girlfriend will be unhappy to learn this and I am sure many users who feel the same as her about r/dangerouslycute's content will feel the same way. However, the scores we found through these models across different parameters does not indicate a lost cause. With all of the assumptions made for our models and with all of the considerations for error made with our models, there is room for improvement. With only 5000 submissions total which were possibly not very scrutinized, our model successfully recognizes from the titles alone which subreddit some submission comes from with ~84% accuracy (doesn't sound bad when thinking about it in this way). This is certainly not app ready, but considerations for the future, we could incorporate:

- ensemble modeling methods to get better model outputs
- pulling more data in general
- analyzing more scrutinzed data
- using a web server to run calculations and many models 
- considering other fetaures to better our model's performance (NSFW calssifiers, numeric popularity weights, text features, image features)

All this considered, we will be continuing to work on a better model to help make our app ready for deployment. Once a successful model is considered, we could begin to further generalize our filtering algorithms and maybe move our model onto other subreddits facing the same problem. 

## Sources and References
- [r/aww subreddit page](https://www.reddit.com/r/aww/)
- [r/natureismetal page](https://www.reddit.com/r/natureismetal)
- [Google Search of Reddit Creation](https://www.google.com/search?client=safari&rls=en&q=when+did+reddit+begin&ie=UTF-8&oe=UTF-8)
- [Reddit Search of Reddit Founders](https://www.reddit.com/r/AskReddit/comments/21875u/what_happend_with_the_guys_who_created_reddit_are/)
- [Reddit Robots.txt Page](https://www.reddit.com/robots.txt)
- [pushshift.io API Webpage](https://pushshift.io)
- [pushift Github Repository](https://github.com/pushshift/api)
- [r/aww Rules & Guidelines](https://www.reddit.com/r/aww/wiki/index)
- [r/natureismetal Rules & Guidelines](https://www.reddit.com/r/natureismetal/submit)
- [Most Common English Words](https://www.rypeapp.com/most-common-english-words/)
- [Wikipedia Search of Most Common Words Shown in Every Language](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists)
- [Scikit-learn Documentation Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Multinomial Naive Bayes forum on meaning of alpha](https://datascience.stackexchange.com/questions/30473/how-does-the-mutlinomial-bayess-alpha-parameter-affects-the-text-classificati)