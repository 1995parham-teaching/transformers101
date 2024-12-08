from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

    tokens = tokenizer("Hello World", return_tensors="pt")

    for t in tokens.input_ids[0]:
        print(tokenizer.decode(t))
