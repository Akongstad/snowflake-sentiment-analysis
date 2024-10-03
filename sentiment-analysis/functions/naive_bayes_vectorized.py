import re

# from _snowflake import vectorized # type: ignore
import pandas as pd

class SnowflakeSentimentVectorized:

    # @vectorized(input = pd.DataFrame)
    def end_partition(self, df):
        probs = self.train(df)
        return self.predict_all(probs, df)
        
    def predict_all(self, probabilities: pd.DataFrame, test_reviews: pd.DataFrame) -> float:
        """predict sentiment of test reviews"""
        # clean
        test_reviews["REVIEW"] = test_reviews["REVIEW"].apply(self._clean_py)
        test_reviews["PREDICTION"] = test_reviews["REVIEW"].apply(self._predict, args=(probabilities,))
        
        accuracy = sum(test_reviews["LABEL"] == test_reviews["PREDICTION"]) / len(test_reviews)
        return accuracy
     
    def _predict(self, review: str, probs: pd.DataFrame) -> int:   
        """predict sentiment of a single review"""
        return 0

    def train(self, df) -> pd.DataFrame:
        """learn probabilities"""
        # clean
        result = df[df["LABEL"].isin([0, 4])]
        result["REVIEW"] = result["REVIEW"].apply(self._clean_py)

        # priors
        smoothing = 1

        p_0 = len(result[result["LABEL"] == 0]) / len(result)
        p_4 = len(result[result["LABEL"] == 4]) / len(result)
        assert p_0 + p_4 == 1

        vocabulary = set()
        for review in result["REVIEW"]:
            vocabulary.update(review)

        # token occurence per class [0,4]
        token_occurence = self._compute_token_occurences(result)
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
        
        return training_results
        

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


import yelp_samples

if "__main__" == __name__:
    samples = yelp_samples.SAMPLES
    sentiment = SnowflakeSentimentVectorized()
    df = pd.DataFrame(samples)
    result = sentiment.end_partition(df)
    print(result)
