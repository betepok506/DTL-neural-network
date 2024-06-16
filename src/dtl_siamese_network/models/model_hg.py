from transformers import ViTImageProcessor, ViTModel


class Model:
    def __init__(self, name_model: str, device='cpu'):
        self.device = device
        self.name_model = name_model
        self.model = self._get_model(name_model)
        self.processor = self._get_processor(name_model)
        pass

    def _get_model(self, name_model):
        if 'google/vit-base-patch16-224-in21k' == name_model:
            return ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
        raise NotImplementedError

    def _get_processor(self, name_model):
        if 'google/vit-base-patch16-224-in21k' == name_model:
            return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        raise NotImplementedError

    def predict(self, image):
        if 'google/vit-base-patch16-224-in21k' == self.name_model:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            return last_hidden_states

