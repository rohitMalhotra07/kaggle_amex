class Config:
    base_dir = './'
    data_dir = 'amex-default-prediction/'
    prc_data_dir = 'processed_data/data/'
    
    feat_dim = 188
    embed_dim = 128  # Embedding size for attention
    num_heads = 4  # Number of attention heads
    # ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    ff_dim = 32
    dropout_rate = 0.3
    num_blocks = 2
    learning_rate=0.001
    LR_START = 1e-6
    LR_MAX = 1e-3
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    EPOCHS = 50
    batch_size=512
    
    model_name = 'mult_attention_conv1d_lr_scheduler_es'
    version = 'v1'
    es_patience = 8
    num_layers_rnn = 2