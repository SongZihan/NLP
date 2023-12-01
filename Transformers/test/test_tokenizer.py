from transformers import BertTokenizerFast

# 加载预训练的分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# 示例文本
text = "Hello, world!"

token = tokenizer.tokenize(text)

print(f"token is: {token}")
ids = tokenizer.convert_tokens_to_ids(token)
print(f"ids is: {ids}")

ids_encode = tokenizer.encode(text,add_special_tokens=False,max_length=512,padding='max_length',truncation=True)
print(f"ids_encode is:{ids_encode}")
token_encoded = tokenizer.convert_ids_to_tokens(ids_encode)
print(f"token_encoded is:{token_encoded}")

