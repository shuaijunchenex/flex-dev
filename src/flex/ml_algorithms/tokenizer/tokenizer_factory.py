
class TokenizerFactory:
    @staticmethod
    def create_tokenizer(tokenizer_type, tokenizer_args):
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type == "simple" or tokenizer_type == "basic_english":
            from .impl.basic_english_tokenizer import BasicEnglishTokenizer
            return BasicEnglishTokenizer().create(tokenizer_args)
        elif tokenizer_type == "bert":
            from .impl.bert_tokenizer import BertTokenizer
            return BertTokenizer().create(tokenizer_args)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")