# learn_torch
Learn Pytorch from the basics.

1. `masked_scatter` : replace `input_embed` vision token holder with  `image_embed`。  

    ```python
    image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                ) # create mask based on input_ids
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    ```