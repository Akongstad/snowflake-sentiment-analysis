-- Benchmarking
create schema bayes_sql_benchmark;

CREATE TABLE benchmark_results (
    language text, 
    dataset text,
    accuracy double, 
    warehouse_size text,
    repetition_num integer,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    elapsed_time_ms NUMBER
);

use warehouse chipmunk_wh_xs;
use warehouse chipmunk_wh_s;
use warehouse chipmunk_wh_m;
use warehouse chipmunk_wh_l;


use schema public;
ALTER SESSION SET USE_CACHED_RESULT=FALSE;
set warehouse_size = (SELECT CURRENT_WAREHOUSE());
set repetition_num = 3;
set language = 'sql';
set dataset = 'yelp-reviews';
SET start_time = CURRENT_TIMESTAMP();


-- Implementation
-- Reading data.
create or replace table yelp_reviews (
    id int IDENTITY(1,1) not null,
    label int not null, 
    review variant not null);
     
COPY INTO yelp_training from @chipmunk_stage/data/train-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;

    
-- Clean data. extract labels
insert into yelp_reviews (label, review) 
    with flattened as (
        select 
            a.data,
            b.key,
            b.value
        from 
            yelp_training a, lateral flatten(input => a.data) b
    )
    select -- min ensure we get a single value. Aggregate
        min(CASE WHEN key LIKE '%label%' THEN value END) AS label,
        clean(min(CASE WHEN key LIKE '%text%' THEN value END)) AS text
    FROM flattened
    GROUP BY data
    HAVING label IN ('0', '4');

-- select *  from yelp_reviews;

-- data processing reviews. Remove special characters and stopwords
create or replace function clean(review string)
returns ARRAY
language python
runtime_version = '3.11'
handler = 'clean_py'
as
$$
import re
stops = {'we', 'further', "shouldn't", 'won', "that'll", 'from', "hasn't", 'yourselves', 'its', 'shouldn', 'into', 'off', 'it', 'about', 'hasn', 'aren', 'the', "weren't", 'yourself', 'such', 'nor', "don't", 'that', 'm', 'most', 'just', 'some', 'until', 'them', 'what', 'my', 'hers', 'was', 'once', 'both', "needn't", "it's", 'not', "isn't", 'few', 'up', 'himself', 'did', "you've", 'why', 'any', 'below', 'her', 'being', 'didn', 'of', 'between', "you'd", "shan't", 'yours', 'isn', 'your', "you'll", 'he', "wasn't", 'down', "mustn't", 'y', 'd', 'doing', 'in', 'again', 'don', 'were', 'hadn', 'while', 'haven', 'ain', 'more', 'him', 'under', 'against', 'with', 'over', 'by', 's', 'very', 'itself', 'theirs', 'as', 'during', 'wouldn', "mightn't", 're', 'same', 'all', 'than', 'when', 't', 'couldn', 'their', 'how', 'our', 'own', 'for', 'those', 'am', "should've", 'has', 'had', 'i', "won't", 'doesn', 'out', 'through', 'myself', 'will', "aren't", 'ourselves', 'these', "couldn't", 'who', 'weren', 'no', 'or', 'then', "haven't", 'above', "you're", 'so', 'mustn', 'an', 'themselves', 'and', 'there', 'she', 'shan', "wouldn't", 'can', 'herself', 'if', 'where', 'now', "hadn't", 'this', 'mightn', 'his', 'you', 'a', 'they', 'too', 'but', 'to', 'here', 'are', 'ma', 'ours', "she's", 'only', 'needn', "doesn't", 'be', 'll', 'should', 'each', 'at', 've', 'do', 'wasn', 'is', 'me', 'does', 'o', 'before', 'on', 'having', 'other', 'have', "didn't", 'been', 'after', 'because', 'which', 'whom'}

def clean_py(review:str) -> list[str] :
    # Remove all puctuation and newlines
    cleaned_text = re.sub(r'(\\n)+|[^\w\s$]|[$(\d)]+', ' ', review)
    lower = cleaned_text.lower().split()
    filtered = filter(lambda word: word not in stops, lower)
    return list(filtered)
$$;


-- Count word
create or replace view total_word_count as
    select top 1000 id, count(value) as word  from yelp_reviews, lateral flatten(input => yelp_reviews.review) as words
    group by id
    order by id;
select * from total_word_count;

-- Count individual words
create or replace table yelp_word_count as 
    select id,  value as word, count(value) as word_count from yelp_reviews, lateral flatten(input => yelp_reviews.review)
    group by id, value;

-- labeled word bag
create or replace table yelp_word_count_filtered as
    SELECT yelp_word_count.id, yelp_reviews.label,yelp_word_count.word, yelp_word_count.word_count
    FROM yelp_word_count
    JOIN yelp_reviews ON yelp_word_count.id = yelp_reviews.id
    WHERE yelp_reviews.label = 0 OR yelp_reviews.label = 4
    ORDER BY yelp_word_count.id asc, yelp_word_count.word_count desc;


-- Bayes training. Preproccessing Done. Now: Count occurences of tokens in each class (0,4)
create or replace table yelp_occurences_per_class as
    select 
        word, 
        label, 
        sum(word_count) as occurrences 
    from yelp_word_count_filtered
    group by word, label
    order by word asc, sum(word_count) desc;


-- Bayes. Estimate  P(c)  as the relative frequency of each class in the training data.
set p_0 = (select count(case when label = 0 then 1 end) / count(*) from yelp_word_count_filtered);
set p_4 = 1-$p_0;
set smoothing = 1;

-- Bayes. calculate  P(w | c) using Laplace smoothing. Multinomial Naïve Bayes: Learning
create or replace view words_0 as
    select * from yelp_occurences_per_class
    where label = 0;
create or replace view words_4 as
    select * from yelp_occurences_per_class
    where label = 4;
create or replace view vocabulary as
    select distinct(word) from yelp_word_count_filtered;

set v_size = (select count(*) from vocabulary);

-- occurnces of w +1 /
-- total occurences in class + vocab_size * 1
create or replace view prob_word_0 as
SELECT 
   word, 
   occurrences,
   ((occurrences + $smoothing)::double / 
   ( (SELECT SUM(occurrences) FROM words_0) + $v_size * $smoothing)::double) AS prob_0
FROM 
   words_0;

create or replace view prob_word_4 as
SELECT 
   word, 
   occurrences,
   (occurrences + $smoothing)::double / 
   ((SELECT SUM(occurrences) FROM words_4) + $v_size * $smoothing)::double AS prob_4,
FROM 
   words_4;

create or replace table yelp_training_results as
select
    COALESCE(prob_word_0.word, prob_word_4.word) AS word,
    COALESCE(prob_word_0.occurrences, 0) AS occurrences_0,
    COALESCE(prob_word_4.occurrences, 0) AS occurrences_4,
    COALESCE(prob_word_0.prob_0, (($smoothing)::double / ((SELECT SUM(occurrences) FROM words_0) + $v_size * $smoothing)::double)) AS prob_0,
    COALESCE(prob_word_4.prob_4, (($smoothing)::double / ((SELECT SUM(occurrences) FROM words_4) + $v_size * $smoothing)::double)) AS prob_4 
    from prob_word_0
    full join prob_word_4 on prob_word_0.word = prob_word_4.word;

select * from yelp_training_results;

-- Bayes classification. Step 1. Read and Clean test data.
create or replace table yelp_test (data variant);
COPY INTO yelp_test from @chipmunk_stage/data/test-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;

create or replace table test_yelp_reviews (
    id int IDENTITY(1,1) not null,
    label int not null, 
    review variant not null);

insert into test_yelp_reviews (label, review) 
    WITH flattened AS (
        SELECT 
            a.data,
            b.key,
            b.value
        FROM 
            yelp_test a, LATERAL FLATTEN(input => a.data) b
    )
    SELECT -- min ensure we get a single value. Aggregate
        min(CASE WHEN key LIKE '%label%' THEN value END) AS label,
        clean(min(CASE WHEN key LIKE '%text%' THEN value END)) AS text
    FROM flattened
    GROUP BY data
    HAVING label IN ('0', '4');
      

-- Remove unknown words 
-- Sentiment analysis -- Scoring. Compute posterior prob from prior and likelyhoods for each class. Choose the highest value.
create or replace table test_yelp_reviews_pred as
WITH flattened AS (
    SELECT
        id,
        label,
        value as review_word
    FROM
        test_yelp_reviews,
        LATERAL FLATTEN(input => review)
),
known_word_reviews AS (
    SELECT
        id,
        label,
        review_word AS word
    FROM
        flattened f
    JOIN
        vocabulary
    ON
        f.review_word = vocabulary.word
), --- preds
preds AS (
    select
        id,
        label,
        revs.word,
        yelp_training_results.prob_0 as pred_0,
        yelp_training_results.prob_4 as pred_4,
    from 
        known_word_reviews revs
    join yelp_training_results on revs.word = yelp_training_results.word
)

SELECT 
    id,
    label,
    ARRAY_AGG(word) as review,
    --Probabilities are typically very small numbers, and multiplying many small numbers can lead to numerical underflow.
    -- The natural logarithm has the property:\log(a * b) = \log(a) + \log(b)
    -- EXP to go back to the original scale
    (LN($p_0) + (SUM(LN(pred_0)))) AS log_pred_0,
    (LN($p_4) + (SUM(LN(pred_4)))) AS log_pred_4
FROM
    preds
GROUP BY
    id,
    label;

create or replace table yelp_eval as
select 
    id, 
    label, 
    case when log_pred_0 > log_pred_4 then 0 else 4 end as pred_label, 
    review, 
    log_pred_0, 
    log_pred_4
from test_yelp_reviews_pred;

create or replace view accuracy as
with correct as (
    select id
    from yelp_eval
    where label = pred_label
)
select 
    (count(correct.id) / count(yelp_eval.id)) as accuracy
from yelp_eval
LEFT JOIN correct ON yelp_eval.id = correct.id;




-- More benchmarking
set accuracy = (select accuracy from accuracy);

SET end_time = CURRENT_TIMESTAMP();
set elapsed_time = (select datediff('milliseconds', $start_time, $end_time));

use schema bayes_sql_benchmark;
INSERT INTO benchmark_results (language, dataset,accuracy, warehouse_size, repetition_num, start_time, end_time, elapsed_time_ms)
VALUES ($language, $dataset, $accuracy, $warehouse_size, $repetition_num, $start_time, $end_time, $elapsed_time);

select * from benchmark_results order by end_time desc;