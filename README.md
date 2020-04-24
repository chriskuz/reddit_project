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