import torch

class ParameterStorage:
    data_path = "../Data/nmt/simplest_eng_fra.csv"
    vocab_save_path="./vocab.pkl"
    storage_path = "./processed_data.pkl"

    model_state_file = "./model/37_model.pt"
    test_performance_file = "./model/performance.json"
    history_file = "./model/training_history.json"

    max_sentence_length = 16 # 最大句子长度

    batch_size = 512
    epochs = 100

    d_model = 32  # 字 Embedding 的维度
    d_ff = 512  # 前向传播隐藏层维度
    d_k = d_v = 32  # K(=Q), V的维度
    n_layers = 6  # 有多少个encoder和decoder
    n_heads = 8  # Multi-Head Attention设置为8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Is_early_stop = True
    early_stop_patience = 8
    learning_rate = 1e-3

    # src_vocab_size = tokenizer.vocab_size
