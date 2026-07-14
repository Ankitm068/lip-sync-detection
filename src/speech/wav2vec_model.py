from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Processor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Wav2VecModel:

    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
    ):

        logger.info("Loading wav2vec 2.0...")

        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name
        )

        self.model = Wav2Vec2Model.from_pretrained(
            model_name
        )

        self.model.eval()

        logger.info("Model Loaded Successfully")