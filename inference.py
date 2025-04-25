from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

from ElectraSpacer.preprocessor import Preprocessor
from ElectraSpacer.dataset import CustomDataset
from ElectraSpacer.trainer import SpacingTrainer
from ElectraSpacer.utils import post_process_function

import time

tokenizer = AutoTokenizer.from_pretrained("monologg/kocharelectra-small-discriminator")
model = AutoModelForTokenClassification.from_pretrained(
    "./ElectraSpacer/models",
    from_tf=bool(".ckpt" in "./ElectraSpacer/models"),
    num_labels=4,
)

class _ElectraSpacer:
    def __init__(self, max_len=256, output_dir="./output"):
        start = time.time()
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.preprocessor = Preprocessor(self.max_len, self.tokenizer)

        training_args = TrainingArguments(output_dir=output_dir)
        trainer = SpacingTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            post_process_function=post_process_function,
        )
        
        self.training_args = training_args
        self.trainer = trainer

    def predict_sentence(self, sent: str) -> str:
        test_dataset = CustomDataset(
            sentence=sent, transform=self.preprocessor.get_input_features
        )

        prediction = self.trainer.predict(
            test_dataset=test_dataset, test_examples=test_dataset
        )

        return prediction["text_prediction"][0]

    def __call__(self, sent: str) -> str:
        sents = sent.split(",")
        new_sent = [sents[0]]
        idx = 0
        for sent in sents[1:]:
            if len(new_sent[idx] + sent) < 100:
                new_sent[idx] = new_sent[idx] + ", " + sent
            else:
                new_sent.append(sent)
                idx += 1

        pred_sent = ", ".join(
            [self.predict_sentence(sent.strip()) for sent in new_sent]
        )

        return pred_sent

    def split_sentence(self, full_sent: str):
        split_sent = full_sent.strip().split(".")
        print(split_sent)
        return (
            ". ".join(self(sent.strip()) for sent in split_sent if sent != "").strip()
            + "."
        )
