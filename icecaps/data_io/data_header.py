class DataHeader:
    def __init__(self, name, data_type, vocab_file=None, vocab_mode="read"):
        self.name = name
        self.data_type = data_type
        self.vocab_file = vocab_file
        self.vocab_mode = vocab_mode
