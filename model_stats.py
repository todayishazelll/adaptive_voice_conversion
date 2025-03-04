import time
from torchinfo import summary
from model import AE

# Start measuring time
start_time = time.time()

# Example input dimensions (Batch size, Channels, Length)
input_size = (1, 1, 128)  # Adjust based on your data
config = {
    'SpeakerEncoder': {
        'c_in': 1, 'c_h': 128, 'c_out': 256, 'kernel_size': 5,
        'bank_size': 8, 'bank_scale': 1, 'c_bank': 128,
        'n_conv_blocks': 6, 'n_dense_blocks': 3,
        'subsample': [2, 1, 2, 1, 2, 1], 'act': 'relu',
        'dropout_rate': 0.5
    },
    'ContentEncoder': {
        'c_in': 1, 'c_h': 128, 'c_out': 256, 'kernel_size': 5,
        'bank_size': 8, 'bank_scale': 1, 'c_bank': 128,
        'n_conv_blocks': 6, 'subsample': [2, 1, 2, 1, 2, 1],
        'act': 'relu', 'dropout_rate': 0.5
    },
    'Decoder': {
        'c_in': 256, 'c_cond': 256, 'c_h': 128, 'c_out': 1,
        'kernel_size': 5, 'n_conv_blocks': 6,
        'upsample': [2, 1, 2, 1, 2, 1], 'act': 'relu',
        'sn': False, 'dropout_rate': 0.5
    }
}

# Initialize model
model = AE(config)

# Whole model summary
print("=== AE Model Summary ===")
summary(model, input_size=input_size)

# SpeakerEncoder summary
print("\n=== SpeakerEncoder Summary ===")
summary(model.speaker_encoder, input_size=input_size)

# ContentEncoder summary
print("\n=== ContentEncoder Summary ===")
summary(model.content_encoder, input_size=input_size)

# Decoder summary (for latent vector and speaker embedding)
latent_input_size = (1, 256, 32)  # Adjust dimensions based on your model's latent space
speaker_input_size = (1, 256)

print("\n=== Decoder Summary ===")
summary(model.decoder, input_size=[latent_input_size, speaker_input_size])

# End measuring time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nTotal Running Time: {elapsed_time:.2f} seconds")
