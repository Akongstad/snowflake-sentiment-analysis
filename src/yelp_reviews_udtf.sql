-- Benchmark
use schema bayes_sql_benchmark;
use warehouse chipmunk_wh_xs;
use warehouse chipmunk_wh_s;
use warehouse chipmunk_wh_m;
use warehouse chipmunk_wh_l;

use schema public;
alter session set USE_CACHED_RESULT=FALSE;
set warehouse_size = (select CURRENT_WAREHOUSE());
set repetition_num = 3;
set language = 'python-udtf';
set dataset = 'yelp-reviews';
set start_time = CURRENT_TIMESTAMP();



-- Implementation
create or replace table yelp_training_udft (data variant);
create or replace table yelp_test_udft (data variant);
copy into yelp_training_udft from @chipmunk_stage/data/train-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;
copy into yelp_test_udft from @chipmunk_stage/data/test-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;

create or replace table yelp_all_udft (data variant, dataset string);

insert into yelp_all_udft (data, dataset)
    select data, 'train' from yelp_training_udft;
insert into yelp_all_udft (data, dataset)
    select data, 'test' from yelp_test_udft;

select * from yelp_all_udft;



create or replace function snowflake_sentiment_vectorized(label int, review string, dataset string)
returns table (id int, label int, review array, dataset string, prediction int)
language python
runtime_version=3.11
packages=('pandas')
handler='SnowflakeSentimentVectorized'
as $$
import re
import math

from _snowflake import vectorized # type: ignore
import pandas as pd

class SnowflakeSentimentVectorized:

    @vectorized(input = pd.DataFrame)
    def end_partition(self, df:pd.DataFrame):
        # clean
        result = df[df["LABEL"].isin([0, 4])]
        result["REVIEW"] = result["REVIEW"].apply(self._clean_py)

        # split
        df_train = result[result["DATASET"] == "train"].copy()
        df_test = result[result["DATASET"] == "test"].copy()

        probs, p_0, p_4, vocabulary = self.train(df_train)
        test_results = self.predict_all(df_test, probs, p_0, p_4, vocabulary)

        # Add ids
        test_results.insert(0, 'ID', test_results.index+1)

        return test_results

    def predict_all(self, test_reviews: pd.DataFrame, probabilities: pd.DataFrame, p_0:float, p_4:float,vocabulary: set[str] ) -> pd.DataFrame:
        """predict sentiment of test reviews"""

        test_reviews["PREDICTION"] = -1
        test_reviews["PREDICTION"] = test_reviews["REVIEW"].apply(self._predict, args=(probabilities, p_0, p_4,vocabulary))

        return test_reviews

    def _predict(self, review: list[str], probabilities: pd.DataFrame, p_0, p_4, vocabulary: set[str]) -> int:
        """predict sentiment of a single review"""
        prob_0, prob_4 = math.log(p_0), math.log(p_4)

        for word in review:
            if word in vocabulary:
                prob_0 += math.log(probabilities.at[word, "PROB_0"])
                prob_4 += math.log(probabilities.at[word, "PROB_4"])

        return 0  if prob_0 > prob_4 else 4

    def train(self, df_train) -> tuple[pd.DataFrame, float, float, set[str]]:
        """learn probabilities"""
        # priors
        smoothing = 1

        p_0 = len(df_train[df_train["LABEL"] == 0]) / len(df_train)
        p_4 = len(df_train[df_train["LABEL"] == 4]) / len(df_train)
        assert p_0 + p_4 == 1

        vocabulary = set()
        for review in df_train["REVIEW"]:
            vocabulary.update(review)

        # token occurence per class [0,4]
        token_occurence = self._compute_token_occurences(df_train)
        # token probability per class [0,4]
        token_prob = self._compute_token_prob(token_occurence, smoothing, vocabulary)

        # token in class_1, but not in class_2. Avoid 0 probs.
        default_prob = smoothing / (sum(token_occurence[0].values()) + len(vocabulary))

        training_results = pd.DataFrame(
            {
                "WORD": list(vocabulary),
                "PROB_0": [token_prob[0][word] if token_prob[0][word] else default_prob for word in vocabulary],
                "PROB_4": [token_prob[4][word] if token_prob[4][word] else default_prob for word in vocabulary],
            }
        )

        return training_results.set_index("WORD"), p_0, p_4, vocabulary


    def _compute_token_prob(
        self,
        token_occurence: dict[int, dict[str, int]],
        smoothing: int,
        vocabulary: set[str],
    ) -> dict[int, dict[str, float]]:
        """token probability per classes. Example: token_prob[0]["word"] = 0.0001"""
        token_prob: dict[int, dict[str, float]] = {0: {}, 4: {}}

        # occurnces of w + 1 /
        # total occurences of words in class + vocab_size * 1
        # remember to update both to avoid 0
        for label in [0, 4]:
            total = sum(token_occurence[label].values())
            for word in vocabulary:
                token_prob[label][word] = (
                    token_occurence[label].get(word, 0) + smoothing
                ) / (total + len(vocabulary)*smoothing)
        return token_prob

    def _compute_token_occurences(self, df) -> dict[int, dict[str, int]]:
        """token occurence per class. Example: token_occurence[0]["word"] = 10"""
        token_occurence: dict[int, dict[str, int]] = {0: {}, 4: {}}

        for _, row in df.iterrows():
            label = row["LABEL"]
            review = row["REVIEW"]
            for word in review:
                token_occurence[label][word] = token_occurence[label].get(word, 0) + 1
        return token_occurence

    def _clean_py(self, review: str) -> list[str]:
        stops = {"we", "further", "shouldn't", "won", "that'll", "from", "hasn't", "yourselves", "its", "shouldn", "into", "off", "it", "about", "hasn", "aren", "the", "weren't", "yourself", "such", "nor", "don't", "that", "m", "most", "just", "some", "until", "them", "what", "my", "hers", "was", "once", "both", "needn't", "it's", "not", "isn't", "few", "up", "himself", "did", "you've", "why", "any", "below", "her", "being", "didn", "of", "between", "you'd", "shan't", "yours", "isn", "your", "you'll", "he", "wasn't", "down", "mustn't", "y", "d", "doing", "in", "again", "don", "were", "hadn", "while", "haven", "ain", "more", "him", "under", "against", "with", "over", "by", "s", "very", "itself", "theirs", "as", "during", "wouldn", "mightn't", "re", "same", "all", "than", "when", "t", "couldn", "their", "how", "our", "own", "for", "those", "am", "should've", "has", "had", "i", "won't", "doesn", "out", "through", "myself", "will", "aren't", "ourselves", "these", "couldn't", "who", "weren", "no", "or", "then", "haven't", "above", "you're", "so", "mustn", "an", "themselves", "and", "there", "she", "shan", "wouldn't", "can", "herself", "if", "where", "now", "hadn't", "this", "mightn", "his", "you", "a", "they", "too", "but", "to", "here", "are", "ma", "ours", "she's", "only", "needn", "doesn't", "be", "ll", "should", "each", "at", "ve", "do", "wasn", "is", "me", "does", "o", "before", "on", "having", "other", "have", "didn't", "been", "after", "because", "which", "whom"}
        # Remove all puctuation and newlines
        cleaned_text = re.sub(r"(\\n)+|[^\w\s$]|[$(\d)]+", " ", review)
        lower = cleaned_text.lower().split()
        filtered = filter(lambda word: word not in stops, lower)
        return list(filtered)

$$;
create or replace table udtf_training_results as
select t.id, t.label, t.prediction, t.review, t.dataset from yelp_all_udft, table(snowflake_sentiment_vectorized(data:label::int, data:text::string, dataset::string)  over (partition by 1)) t;

select * from udtf_training_results;

create or replace view accuracy as
with correct as (
    select id
    from udtf_training_results
    where label = prediction
)
select
    (count(correct.id) / count(udtf_training_results.id)) as accuracy
    from udtf_training_results
    left JOIN correct ON udtf_training_results.id = correct.id;

//select * from accuracy;


-- More benchmarking
set accuracy = (select accuracy from accuracy);

set end_time = CURRENT_TIMESTAMP();
set elapsed_time = (select datediff('milliseconds', $start_time, $end_time));

use schema bayes_sql_benchmark;
insert into benchmark_results (language, dataset,accuracy, warehouse_size, repetition_num, start_time, end_time, elapsed_time_ms)
values ($language, $dataset, $accuracy, $warehouse_size, $repetition_num, $start_time, $end_time, $elapsed_time);

select * from benchmark_results order by end_time desc;
