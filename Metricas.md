| Idioma | Preprocesamiento | Extractor de features | Clasificador | MCC   | Precision | Recall | F1 Score - Critical | F1 Score - Conspiracy | Macro F1-Score |
|:------:|:----------------:| --------------------- |:------------:| ----- | --------- | ------ | ------------------- | --------------------- | -------------- |
|   ES   |      Basic       | CountVectorizer       |     MNB      | 0.626 | 0.808     | 0.818  | 0.856               | 0.768                 | 0.812          |
|   ES   |      Spacy       | CountVectorizer       |     MNB      | 0.611 | 0.804     | 0.807  | 0.856               | 0.755                 | 0.805          |
|   ES   |    Spacy_pos     | CountVectorizer       |     MNB      | 0.637 | 0.818     | 0.819  | 0.867               | 0.77                  | 0.818          |
|   ES   |    Spacy_pos     | TF-IDF                |      LR      | 0.619 | 0.838     | 0.784  | 0.873               | 0.725                 | 0.799          |
|   ES   |    Spacy_pos     | TF-IDF                |     LRCV     | 0.677 | 0.856     | 0.822  | 0.889               | 0.779                 | 0.834          |
|   ES   |    Spacy_pos     | W2V                   |     LRCV     | 0.63  | 0.825     | 0.806  | 0.871               | 0.755                 | 0.813          |
|   ES   |    Spacy_pos     | W2V                   |      RF      | 0.587 | 0.83      | 0.761  |                     |                       |                |
|   EN   |      Basic       | CountVectorizer       |     MNB      | 0.658 | 0.831     | 0.828  | 0.883               | 0.775                 | 0.829          |
|   EN   |      Spacy       | CountVectorizer       |     MNB      | 0.618 | 0.82      | 0.798  | 0.875               | 0.739                 | 0.807          |
|   EN   |    Spacy_pos     | CountVectorizer       |     MNB      | 0.65  | 0.836     | 0.814  | 0.885               | 0.761                 | 0.823          |
|   EN   |    Spacy_pos     | TF-IDF                |      LR      | 0.626 | 0.866     | 0.768  | 0.884               | 0.698                 | 0.791          |
|   EN   |    Spacy_pos     | TF-IDF                |     LRCV     | 0.678 | 0.863     | 0.817  | 0.897               | 0.768                 | 0.833          |
|   EN   |    Spacy_pos     | W2V                   |     LRCV     | 0.583 | 0.817     | 0.768  | 0.869               | 0.697                 | 0.783          |
