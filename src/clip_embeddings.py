import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor


class ClipEncoder:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32"
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Using device:", self.device)

        self.model = CLIPModel.from_pretrained(
            model_name
        )

        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()


    def encode_images(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(self.device)

        with torch.no_grad():

            # Run image through CLIP vision model
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values
            )

            # Get pooled image representation
            pooled_output = vision_outputs.pooler_output

            # Project into CLIP shared embedding space
            embeddings = self.model.visual_projection(
                pooled_output
            )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype("float32")
        )

        # Normalize embeddings
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        embeddings = embeddings / norms

        return embeddings


    def encode_texts(self, texts):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs[
                    "attention_mask"
                ]
            )

            pooled_output = (
                text_outputs.pooler_output
            )

            embeddings = (
                self.model.text_projection(
                    pooled_output
                )
            )

        embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype("float32")
        )

        # Normalize embeddings
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        embeddings = embeddings / norms

        return embeddings