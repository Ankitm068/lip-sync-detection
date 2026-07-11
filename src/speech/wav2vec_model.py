from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Processor


class Wav2VecModel:

    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
    ):

        print("Loading wav2vec 2.0...")

        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name
        )

        self.model = Wav2Vec2Model.from_pretrained(
            model_name
        )

        self.model.eval()

        print("Model Loaded Successfully")