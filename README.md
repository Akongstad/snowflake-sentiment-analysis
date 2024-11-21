# snowflake-sentiment-analysis

Mastering Snowflake: Sentiment Analysis and Performance Experiments | Advanced Datasystems 2024 | ITU

This repository contains:

- 2 implementations of naive bayes sentiment analysis. (Includes the benchmarking code.)
  1. An sql implementation in `yelp_reviews.sql`. Runs in Snowflake.
  2. A python UDTF implementation that takes the yelp review data as a Pandas Dataframe and returns a table with predictions when run with `yelp_reviews_udtf.sql`. Runs in snowflake.

- A TPC-H benchmark using queries 1, 5, and 18 in `sentiment-analysis/tpch_benchmarks.sql`

- All results as CSV files in `results`
