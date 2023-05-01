# AI Sentiment Dataset
This repository contains a dataset of social media posts about artificial
intelligence along with the date, platform, username, and country code where
available. The data comes from YouTube (66.6%, comments), Reddit (28.5%, comments),
Twitter (4.9%, tweets). It also contains all the source code that was used
for gathering the data, as well as useful visualizations and analysis which
pulls insights from this data.

This work was done as group coursework for the Applied Data Science unit
at the University of Bristol.

## Dataset
The dataset located in `AI_sentiment_dataset.csv` contains the following attributes:
| Attribute | Description |
| --- | --- |
| Content | The text of the social media post/comment |
| User | Username of the person who posted it |
| Date | Date the post/comment was posted. Format: YYYY-MM-DD |
| Reactions | Number of likes/upvotes. Can be negative for Reddit data. |
| N_Children | Number of replies to the post/comment. Note that the dataset does not include replies |
| Post Title | Title of the YouTube video/Reddit post the comment is under |
| Platform | One of `['YouTube', 'Reddit', 'Twitter']` |
| Country | 3-letter country code following [ISO 3166 Alpha-3](https://www.iban.com/country-codes) |

## Source code
The code used to gather the data is in `src/gathering`. To present findings
from the dataset, we used an ensemble of sentiment analysis models. This
analysis can be found in `src/analysis`. We then visualize our findings
in a series of Jupyter notebooks in `src/visualization`. The charts this
generated are also present in `src/visualization/figures`.

Data in various stages of being gathered and analyzed is in `data`.
