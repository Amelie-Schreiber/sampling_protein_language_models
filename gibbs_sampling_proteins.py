import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import numpy as np

def gibbs_sampling_likeliest_with_random_masking(sequence, mask_percentage, iterations):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model.eval()  # Set model to evaluation mode
    
    for _ in range(iterations):
        # Convert sequence to tokens and identify positions to mask
        inputs = tokenizer(sequence, return_tensors="pt")
        total_tokens = inputs["input_ids"].shape[1]
        mask_count = int(total_tokens * mask_percentage)
        mask_positions = np.random.choice(total_tokens, mask_count, replace=False)
        
        # Create a masked copy of the sequence
        masked_inputs = inputs["input_ids"].clone()
        for pos in mask_positions:
            masked_inputs[0][pos] = tokenizer.mask_token_id
        
        # Predict the masked tokens
        with torch.no_grad():
            logits = model(input_ids=masked_inputs).logits
        
        # Replace the masked tokens with predicted tokens
        for pos in mask_positions:
            predicted_token_id = logits[0, pos].argmax(axis=-1).item()
            inputs["input_ids"][0][pos] = predicted_token_id
        
        # Convert token IDs back to sequence
        sequence = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True).replace(" ", "")

    return sequence

# Example usage
# protein_sequence = "MNSVTVSHAPYTITYHDDWEPVMSQLVEFYNEVASWLLRDETSPIPDKFFIQLKQPLRNKRVCVCGIDPYPKDGTGVPFESPNFTKKSIKEIASSISRLTGVIDYKGYNLNIIDGVIPWNYYLSCKLGETKSHAIYWDKISKLLLQHITKHVSVLYCLGKTDFSNIRAKLESPVTTIVGYHPAARDRQFEKDRSFEIINVLLELDNKVPINWAQGFIY"
# masked_percentage = 0.10
# iterations = 10
# mutated_sequence = gibbs_sampling_likeliest_with_random_masking(protein_sequence, masked_percentage, iterations)
# print(mutated_sequence)

def gibbs_sampling_from_distribution_with_random_masking(sequence, mask_percentage, iterations):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model.eval()  # Set model to evaluation mode
    
    for _ in range(iterations):
        # Convert sequence to tokens and identify positions to mask
        inputs = tokenizer(sequence, return_tensors="pt")
        total_tokens = inputs["input_ids"].shape[1]
        mask_count = int(total_tokens * mask_percentage)
        mask_positions = np.random.choice(total_tokens, mask_count, replace=False)
        
        # Create a masked copy of the sequence
        masked_inputs = inputs["input_ids"].clone()
        for pos in mask_positions:
            masked_inputs[0][pos] = tokenizer.mask_token_id
        
        # Predict the masked tokens
        with torch.no_grad():
            logits = model(input_ids=masked_inputs).logits
        
        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Replace the masked tokens by sampling from the predicted distribution
        for pos in mask_positions:
            predicted_token_id = torch.multinomial(probabilities[0, pos], 1).item()
            inputs["input_ids"][0][pos] = predicted_token_id
        
        # Convert token IDs back to sequence without spaces
        sequence = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True).replace(" ", "")
    
    return sequence

# Example usage
# protein_sequence = "MNSVTVSHAPYTITYHDDWEPVMSQLVEFYNEVASWLLRDETSPIPDKFFIQLKQPLRNKRVCVCGIDPYPKDGTGVPFESPNFTKKSIKEIASSISRLTGVIDYKGYNLNIIDGVIPWNYYLSCKLGETKSHAIYWDKISKLLLQHITKHVSVLYCLGKTDFSNIRAKLESPVTTIVGYHPAARDRQFEKDRSFEIINVLLELDNKVPINWAQGFIY"
# masked_percentage = 0.10
# iterations = 10
# mutated_sequence = gibbs_sampling_from_distribution_with_random_masking(protein_sequence, masked_percentage, iterations)
# print(mutated_sequence)