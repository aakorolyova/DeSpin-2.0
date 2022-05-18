from transformers import TFAutoModelForTokenClassification


def putorch_to_tf_token_classif(path):
    tf_model = TFAutoModelForTokenClassification.from_pretrained(path, from_pt=True)
    tf_model.save_pretrained(path)


if __name__=='__main__':
    path = 'enter your model path'
    putorch_to_tf_token_classif(path)