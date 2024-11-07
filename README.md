# snowflake-sentiment-analysis

Mastering Snowflake: Sentiment Analysis and Performance Experiments | Advanced Datasystems 2024 | ITU

This repository contains:

- 2 implementations of naive bayes.
  1. An sql implementation in `yelp_reviews.sql` for snowflake
  2. A python UDTF implementation that takes the yelp review data as a Pandas Dataframe and returns a table with predictions when run with `yelp_reviews_udtf.sql` runs in snowflake.

- Benchmarking code.
