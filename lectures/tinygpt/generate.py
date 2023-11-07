import torch


def generate(model, prompt, tokenizer, length=100, device='cpu'):
    model.eval()
    with torch.no_grad():
        input_tensor = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        for i in range(length):
            # Make sure it has the correct length
            while len(input_tensor[0]) < model.config.n_block:
                input_tensor = torch.cat((input_tensor, torch.tensor([tokenizer.padding_idx]).unsqueeze(
                    0).to(device)), 1)  # pad with the padding character
            input_tensor = input_tensor[:, -model.config.n_block:]  # take the last n_block characters

            # Create the key_padding_mask for the input tensor
            key_padding_mask = input_tensor == tokenizer.padding_idx

            last_non_padded_position = (
                input_tensor[0] != tokenizer.padding_idx).nonzero(as_tuple=True)[0][-1]

            # Forward pass
            logits = model(input_tensor, key_padding_mask=key_padding_mask)
            probs = torch.nn.functional.softmax(logits[0, last_non_padded_position, :], dim=-1)

            # Sample next character
            predicted_id = torch.multinomial(probs, num_samples=1)

            input_tensor = input_tensor[:, input_tensor[0] != tokenizer.padding_idx]
            input_tensor = torch.cat((input_tensor, predicted_id.unsqueeze(0)), 1)

            yield input_tensor[0]
