# Matering Snowflake | Project
Mastering Snowflake: Sentiment Analysis and Performance Experiments | Advanced Datasystems 2024 | ITU
## Contents
This repository contains:

- **src/yelp_reviews.sql**: SQL implementation f naive bayes sentiment analysis. (Includes the benchmarking code.)
- **src/yelp_reviews_udtf.sql**: Snowflake Vectorized UDTF implementation of naive bayes sentiment analysis for snowflake. (Includes the benchmarking code.)
- **src/tpch_benchmarks.sql**: A TPC-H benchmark experiment using queries 1, 5, and 18
- **src/functions**: Python only implementation of the Vectorized UDTF and data cleaning function.
- **results/**: Experiment results from the TPCH and sentiment analysis experiments as CSV files
