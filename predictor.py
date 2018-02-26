import urllib

import pandas as pd
from h2o.frame import H2OFrame
from h2o.estimators.word2vec import H2OWord2vecEstimator
import h2o

class Predictor:

    def __init__(self, all_jokes: H2OFrame):
        stop_words_data_path = "https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/h2o-world-2017/nlp/stopwords.csv"
        self.STOP_WORDS = pd.read_csv(stop_words_data_path, header=0)
        self.STOP_WORDS = list(self.STOP_WORDS['STOP_WORD'])
        self.words = self._tokenize(all_jokes["dad_joke"])

    def _init_w2v(self):
        self.w2v_model = H2OWord2vecEstimator(vec_size=100, model_id="w2v.hex")
        self.w2v_model.train(training_frame=self.words)

    def _init_dad_joke_model(self, path_to_model):
        self.dad_joke_model = h2o.load_model(path_to_model)

    def make_prediction(self, text: str) -> float:
        """
        0 - not a dad joke
        1 - dad joke
        """
        joke = H2OFrame([[text]])
        joke.col_names = ["dad_joke"]
        predict_var = self._predict(joke)
        x = predict_var.as_data_frame()
        return x['predict'][0]

    def _predict(self, joke: H2OFrame):
        words = self._tokenize(joke["dad_joke"].ascharacter())
        reviews_vec = self.w2v_model.transform(words, aggregate_method="AVERAGE")

        model_data = joke.cbind(reviews_vec)
        return self.dad_joke_model.predict(model_data)


    def _tokenize(self, sentences):
        tokenized = sentences.tokenize("\\W+")
        tokenized_lower = tokenized.tolower()
        tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
        tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
        tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(self.STOP_WORDS)),:]
        return tokenized_words