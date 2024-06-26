import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline


def add_category_columns(df, labels):
    # TODO Exercise 1: Add one column per label to the dataframe (i.e.
    # you need to add 4 columns, called "CAMPAIGNER", "VICTIM", "AGENT",
    # and "FACILITATOR"): its value should be 1 if there is at least one
    # span annotated with this label in the "annotations" field, and 0
    # if there is no span with this label.
    #
    # This function returns a pandas DataFrame.
    def search_for_annotation(sample, label):
        return any([x["category"] == label for x in sample["annotations"]])

    for label in labels:
        df[label] = df.apply(
            lambda sample: search_for_annotation(sample, label), axis=1
        ).values

    return df


def split_data(df, labels):
    # TODO Exercise 2: Create a train/dev/test split, stratifying by
    # the four labels. You should use 60% for training, 20% for development
    # and 20% for testing. Use 42 as random state.
    #
    # This function returns three pandas DataFrames (one for each split),
    # with the same columns as the input dataframe.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df["id"], df[labels], stratify=df[labels], test_size=0.4, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_tmp, y_tmp, stratify=y_tmp, test_size=0.5, random_state=42
    )
    train_df = df[df["id"].isin(X_train)]
    dev_df = df[df["id"].isin(X_dev)]
    test_df = df[df["id"].isin(X_test)]

    return train_df, dev_df, test_df


def prepare_data_for_labeling(curr_df, labels, label2id, nlp):
    # TODO Exercise 3: Prepare the data for token classification.
    #
    # The format required to fine-tune a model for token classification
    # using transformers is a dictionary per document, with the following
    # key-value pairs:
    # * "id": the id of the document (int or string).
    # * "tokens": the list of tokens in the document (a list).
    # * "tags": the list of tags associated to the tokens (a list of
    #           the same length as "tokens"). Note that this is neither
    #           the label name (e.g. "AGENT") nor the BIO tag (e.g.
    #           "B-AGENT"): it should be the id of the BIO tag (e.g. 2).
    #
    # In this exercise, the inputs of our classifier will be at the level
    # of the sentence (i.e. not the full document). For that, we need to
    # segment documents into sentences. To do this:
    # 1. Convert each document (column "text") into a SpaCy document.
    # 2. Iterate over the sentences of the Doc object (tip: use `.sents`).
    #    Iterate over the tokens in each sentence, keeping the position
    #    of the token within the document (i.e., token.i) and the token
    #    text (i.e., token.text).
    # 3. For each sentence, return a dictionary with the following fields:
    #    "id", "tokens", and "tags":
    #    * The "id" should be a string with the follwing format: "xxxx_yy",
    #      where `xxxx` is the document id and `yy` is the position (within
    #      the document) of the first token in the sentence.
    #    * The "tokens" value should be a list of the tokens in the sentence.
    #    * The "tags" value should be a list of the BIO tags ids, based on the
    #      values of the "annotations" column in the dataframe (note that we
    #      can map tokens to the annotations because we are using the same
    #      SpaCy model that was used by the organisers to tokenise the dataset).
    #
    # This function returns a list of dictionaries, each dictionary consisting
    # of three keys: "id", "tokens" and "tags".
    #
    res = []
    for idx, row in curr_df.iterrows():
        doc = nlp(row["text"])
        sentences = doc.sents

        for sentence in sentences:
            sentence_dict = {
                "id": str(row["id"]) + "_" + str(sentence[0].i),
                "tokens": [],
                "tags": [],
            }
            for token in sentence:
                sentence_dict["tokens"].append(token.text)
                tag = label2id["O"]
                for span in row["annotations"]:
                    if span["start_spacy_token"] == token.i:
                        tag = label2id.get("B-" + span["category"], tag)
                        break
                    elif (
                        span["start_spacy_token"] < token.i
                        and token.i < span["end_spacy_token"]
                    ):
                        tag = label2id.get("I-" + span["category"], tag)
                        break
                sentence_dict["tags"].append(tag)
            res.append(sentence_dict)
    return res


def apply_model(model_name, test_df, nlp):
    # -------------------------------------------------------------
    # TODO Exercise 5: Given the model name and the test set as a dataframe
    # (and the spaCy object for segmenting documents into sentences), this
    # function returns the output as expected for the shared task evaluation
    # (for more information, see:
    # https://github.com/dkorenci/pan-clef-2024-oppositional/blob/main/README-DATA):
    # "For sequence labeling, the output must be a list of dictionaries,
    # each with 'id', 'annotations' fields. The 'annotations' list can
    # either be empty, or it must contain dictionaries with 'category',
    # 'start_char', 'end_char' fields."
    #
    # Tip: You can use transformers pipelines (task "ner") for that:
    # https://huggingface.co/docs/transformers/main_classes/pipelines.
    # In order to group tokens that belong to the same labels, you can
    # use the "aggregation_strategy" parameter.
    #
    # This function returns the test set formatted as a list of dictionries
    # as required by the shared task.
    #Variable to store results
    results = []
    #Ner pipeline
    ner_pipeline = pipeline(
        model=model_name, aggregation_strategy="simple", task="ner"
    )

    for _, row in list(test_df.iterrows()):
        id = row["id"]
        res_dict = {"id": id, "annotations": []}

        doc = nlp(row["text"])
        #Get string for all sentences
        sentences = [sent.text for sent in doc.sents]
        #Pass it over to our pipeline
        results_pipeline = ner_pipeline(sentences)
        #For each result
        for result in results_pipeline:
            #Get the tag
            for sent_tag in result:
                #Ignore "O" tags
                if sent_tag["entity_group"] != 'O':
                    #Aux dict to store data
                    aux_dict = {
                        "category": sent_tag["entity_group"],
                        "start_char": sent_tag["start"],
                        "end_char": sent_tag["end"],
                    }
                    #Add it to temporary dict
                    res_dict["annotations"].append(aux_dict)
        #return data
        results.append(res_dict)

    return results
