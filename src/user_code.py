import os
import sys
import csv
import json
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertJapaneseTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader, TensorDataset

# Logging
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter(
    "%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s"
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

import argparse

parser = argparse.ArgumentParser(description="Use more complex ML model for chatbot")
parser.add_argument(
    "--model",
    dest="model",
    help="Model name",
    default="cl-tohoku/bert-base-japanese-whole-word-masking",
    type=str,
)
parser.add_argument(
    "--train", dest="train", help="Enable training", action="store_true"
)
parser.add_argument("--test", dest="test", help="Enable testing", action="store_true")
parser.add_argument(
    "--epochs",
    dest="epochs",
    help="The number of epochs",
    action="store",
    default=10,
    type=int,
)
parser.add_argument(
    "--test_ratio",
    dest="test_ratio",
    help="Ratio of test data",
    action="store",
    default=0.1,
    type=float,
)
args = parser.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, classes):
        self.encodings = encodings
        self.classes = classes

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.classes[idx]  # 真の値のkeyはlabelsでなければならない
        return item

    def __len__(self):
        return len(self.classes)


class CustomCallback(TrainerCallback):
    def __init__(self, output_dir, logging_steps=10):
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.logs = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.logging_steps == 0 and state.global_step > 0:
            log_item = {
                "step": state.global_step,
                "epoch": state.epoch,
                "learning_rate": logs.get("learning_rate"),
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "eval_runtime": logs.get("eval_runtime"),
                "eval_samples_per_second": logs.get("eval_samples_per_second"),
                "eval_steps_per_second": logs.get("eval_steps_per_second"),
            }

            # None valueは記録しない
            log_item = {k: v for k, v in log_item.items() if v is not None}
            self.logs.append(log_item)
            self._write_logs()

    def _write_logs(self):
        with open(os.path.join(self.output_dir, "training_eval_logs.json"), "w") as f:
            json.dump(self.logs, f, indent=4)


class ChatbotModelEvaluation:
    """
    label: 回答データのUUID
    class: 回答データのUUIDを分類問題で扱いやすいように整数の値変更したもの
    """

    def __init__(
        self,
        model_name: str = "cl-tohoku/bert-base-japanese",
        train_data_file_paths: list[str] = [
            f"{os.path.dirname(__file__)}/train_data_file.csv",
        ],
        validation_data_file_paths: list[str] = [
            f"{os.path.dirname(__file__)}/validation_data_file.csv",
        ],
        test_data_file_paths: str = [
            f"{os.path.dirname(__file__)}/test_data_file.csv",
        ],
    ):
        self.model_name = model_name
        self.train_data_file_paths = train_data_file_paths
        self.validation_data_file_paths = validation_data_file_paths
        self.test_data_file_paths = test_data_file_paths

        self.number_of_classes = 0
        self.label_2_class = {}
        self.class_2_label = {}
        self.tokenizer = None
        self.model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"{self.device} will be used in the following series.")

    def get_class_by_label(self, label: str):
        if not label in self.label_2_class:
            _class = len(self.label_2_class)
            self.label_2_class[label] = _class
        return self.label_2_class[label]

    def set_label_by_class(self):
        self.class_2_label = {value: key for key, value in self.label_2_class.items()}

    def get_label_by_class(self, _class: int):
        return self.class_2_label[_class]

    def _load_data(
        self, file_paths: list[str], element_2_index_map: dict = {"text": 0, "label": 2}
    ):
        texts_already_taken_into_account = []
        texts, classes = [], []
        for i_file_path, file_path in enumerate(file_paths):
            logger.info(
                f"[ {i_file_path+1} / {len(file_paths)} ] Load data from {file_path}"
            )
            with open(file_path, mode="r") as f:
                reader = csv.reader(f)
                for i_row, row in enumerate(reader):
                    if i_row > 0:
                        if (
                            not row[element_2_index_map["text"]]
                            in texts_already_taken_into_account
                        ):
                            texts_already_taken_into_account.append(
                                row[element_2_index_map["text"]]
                            )
                            texts.append(row[element_2_index_map["text"]])
                            classes.append(
                                self.get_class_by_label(
                                    label=row[element_2_index_map["label"]]
                                )
                            )

        return texts, classes

    def load_train_data(self, element_2_index_map: dict = {"text": 0, "label": 2}):
        logger.info(f"Load training data")
        return self._load_data(
            file_paths=self.train_data_file_paths,
            element_2_index_map=element_2_index_map,
        )

    def load_validation_data(self, element_2_index_map: dict = {"text": 0, "label": 2}):
        logger.info(f"Load validation data")
        return self._load_data(
            file_paths=self.validation_data_file_paths,
            element_2_index_map=element_2_index_map,
        )

    def load_test_data(self, element_2_index_map: dict = {"text": 0, "label": 2}):
        logger.info(f"Load test data")
        return self._load_data(
            file_paths=self.test_data_file_paths,
            element_2_index_map=element_2_index_map,
        )

    def set_number_of_unique_classes(self):
        logger.info("Set the number of unique classes")
        self.number_of_classes = len(self.label_2_class)

    def set_tokenizer_and_model(self):
        logger.info(f"Set tokenizer and model for {self.model_name}")
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.number_of_classes
        ).to(self.device)

    def train_validation_split(
        self, train_texts, train_classes, validation_size: float = 0.2
    ):
        logger.info(
            f"Split training data into training ({1 - validation_size}) and validation ({validation_size})"
        )
        train_texts, validation_texts, train_classes, validation_classes = (
            train_test_split(
                train_texts,
                train_classes,
                test_size=validation_size,
                random_state=42,
            )
        )

        return train_texts, validation_texts, train_classes, validation_classes

    def tokenization(
        self,
        train_texts,
        validation_texts,
        truncation: bool = True,
        padding: bool = True,
    ):
        train_encodings = self.tokenizer(
            train_texts,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )
        validation_encodings = self.tokenizer(
            validation_texts,
            truncation=truncation,
            padding=padding,
            return_tensors="pt",
        )

        return train_encodings, validation_encodings

    def convert_into_torch_dataset(
        self, train_encodings, validation_encodings, train_classes, validation_classes
    ):
        for key in train_encodings:
            train_encodings[key] = train_encodings[key].clone().detach().to(self.device)
        for key in validation_encodings:
            validation_encodings[key] = (
                validation_encodings[key].clone().detach().to(self.device)
            )
        train_classes = torch.tensor(train_classes).clone().detach().to(self.device)
        validation_classes = (
            torch.tensor(validation_classes).clone().detach().to(self.device)
        )

        train_dataset = Dataset(train_encodings, train_classes)
        validation_dataset = Dataset(validation_encodings, validation_classes)

        return train_dataset, validation_dataset

    def training(self, train_dataset, validation_dataset):
        output_dir = (
            f"../results/{'/'.join(self.train_data_file_paths[0].split('/')[-2:-1])}"
        )
        training_arguments = TrainingArguments(
            eval_strategy="steps",  # evalデータセットに対し評価を行うタイミング
            eval_steps=100,  # evalデータセットに対し評価を行う間隔
            logging_dir="../logs",  # ログ出力ディレクトリ
            logging_steps=100,  # ロギングの頻度
            num_train_epochs=args.epochs,  # エポック数
            output_dir=output_dir,  # 出力ディレクトリ
            per_device_train_batch_size=16,  # トレーニングバッチサイズ
            per_device_eval_batch_size=16,  # 評価バッチサイズ
            save_strategy="epoch",  # パラメータなどの情報を保存するタイミング
            warmup_steps=500,  # ウォームアップステップ数
            weight_decay=0.01,  # Weight decay
            no_cuda=not torch.cuda.is_available(),  # GPUを使用するかどうかのフラグ
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def testing(
        self,
        test_texts,
        test_classes,
        batch_size: int = 8,
        save_file_path=f"{os.path.dirname(__file__)}/../test_results/test_result.csv",
    ):
        test_encodings = self.tokenizer(
            test_texts, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        test_classes = torch.tensor(test_classes).clone().detach().to(self.device)

        test_dataset = TensorDataset(
            test_encodings["input_ids"], test_encodings["attention_mask"]
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        all_predicted_classes = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Predicting"):
                input_ids, attention_mask = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=1)
                all_predicted_classes.extend(predicted_classes.tolist())

        if not os.path.exists(os.path.dirname(save_file_path)):
            os.makedirs(os.path.dirname(save_file_path))

        with open(save_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "truth", "prediction"])

            for i, (text, truth, prediction) in enumerate(
                zip(test_texts, test_classes, all_predicted_classes)
            ):
                writer.writerow([text, truth.item(), prediction])
                if i % 100 == 0:
                    print(f"Text: {text}, Truth: {truth}, Prediction: {prediction}")

    def save(self, save_directory: str = "./trained_model"):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


if __name__ == "__main__":
    evaluator = ChatbotModelEvaluation(
        model_name=args.model,
        train_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt35_upto3.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt35_upto5.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt35_upto10.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt35_upto15.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt35_upto20.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto3.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto5.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto10.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto15.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto20.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto25.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-002/gen_gpt4_upto0.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-003/gen_gpt4_upto0.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-004/gen_gpt4_upto0.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-005/gen_gpt4_upto0.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-006/gen_gpt4_upto7.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-007/gen_gpt4_upto20.csv",
            # f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-008/gen_gpt4_upto25.csv",
        ],
        validation_data_file_paths=[
            f"{os.path.dirname(__file__)}/../../chatbot-effective-data-creation/data/OPEN-UI/generated/prompt-001/gen_gpt4_upto25.csv",
        ],
        test_data_file_paths=[
            f"{os.path.dirname(__file__)}/../dataset/test/train.csv",
        ],
    )
    train_texts, train_classes = evaluator.load_train_data()
    validation_texts, validation_classes = evaluator.load_validation_data()
    test_texts, test_classes = evaluator.load_test_data()
    evaluator.set_number_of_unique_classes()
    evaluator.set_label_by_class()
    evaluator.set_tokenizer_and_model()
    train_encodings, validation_encodings = evaluator.tokenization(
        train_texts=train_texts, validation_texts=validation_texts
    )
    train_dataset, validation_dataset = evaluator.convert_into_torch_dataset(
        train_encodings=train_encodings,
        validation_encodings=validation_encodings,
        train_classes=train_classes,
        validation_classes=validation_classes,
    )
    if args.train:
        evaluator.training(
            train_dataset=train_dataset, validation_dataset=validation_dataset
        )
    if args.test:
        evaluator.testing(
            test_texts=test_texts,
            test_classes=test_classes,
            batch_size=8,
            save_file_path=f"{os.path.dirname(__file__)}/{args.model}/test_result.csv",
        )
