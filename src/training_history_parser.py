import os
import sys
import csv
import json
import matplotlib.pyplot as plt

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


class TrainingHistoryParser:
    def __init__(
        self,
        result_file_path: str = f"{os.path.dirname(__file__)}/../results/result.json",
    ):
        self.result_file_path = result_file_path
        self.train_loss_histories = {}
        self.eval_loss_histories = {}

    def parse(self):
        json_open = open(self.result_file_path, "r")
        json_load = json.load(json_open)
        log_histories = json_load.get("log_history", {})
        for log_history in log_histories:
            if "loss" in log_history and "step" in log_history:
                self.train_loss_histories[log_history.get("step")] = log_history.get(
                    "loss"
                )
            if "eval_loss" in log_history and "step" in log_history:
                self.eval_loss_histories[log_history.get("step")] = log_history.get(
                    "eval_loss"
                )

    def get_step_vs_loss(
        self, plot_name=f"{os.path.dirname(__file__)}/../deliverables/step_vs_loss.png"
    ):
        if not os.path.exists(os.path.dirname(plot_name)):
            os.makedirs(os.path.dirname(plot_name))

        train_steps, train_losses = (
            self.train_loss_histories.keys(),
            self.train_loss_histories.values(),
        )
        eval_steps, eval_losses = (
            self.eval_loss_histories.keys(),
            self.eval_loss_histories.values(),
        )

        plt.figure()

        plt.title("Step vs Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid()

        plt.plot(train_steps, train_losses, linewidth=2.0, label="Train")
        plt.plot(eval_steps, eval_losses, linewidth=2.0, label="Validation")

        plt.legend()
        plt.savefig(plot_name)
        logger.info(f"{plot_name} was generated.")


class TestResultParser:
    def __init__(
        self,
        test_result_file_path: str = f"{os.path.dirname(__file__)}/../test_results/result.csv",
    ):
        self.test_result_file_path = test_result_file_path
        self.test_result_dict = {}

    def parse(self):
        with open(self.test_result_file_path, mode="r") as f:
            reader = csv.reader(f)
            for i_row, row in enumerate(reader):
                if i_row > 0:
                    text = row[0]
                    truth = row[1]
                    prediction = row[2]
                    self.test_result_dict[text] = {
                        "truth": truth,
                        "prediction": prediction,
                    }

    def _get_truths(self):
        return [self.test_result_dict[text]["truth"] for text in self.test_result_dict]

    def _get_predictions(self):
        return [
            self.test_result_dict[text]["prediction"] for text in self.test_result_dict
        ]

    def get_accuracy(self):
        n_data = 0
        n_correct = 0
        accuracy = 0
        for truth, prediction in zip(self._get_truths(), self._get_predictions()):
            n_data += 1
            if truth == prediction:
                n_correct += 1

        accuracy = float(n_correct) / float(n_data)
        logger.info(f"Accuracy: {accuracy}")
        return accuracy


if __name__ == "__main__":
    parser = TrainingHistoryParser(
        result_file_path=f"{os.path.dirname(__file__)}/../results/checkpoint-42976/trainer_state.json"
    )
    parser.parse()
    parser.get_step_vs_loss(
        plot_name=f"{os.path.dirname(__file__)}/../deliverables/step_vs_loss.png"
    )

    parser = TestResultParser(
        test_result_file_path=f"{os.path.dirname(__file__)}/../test_results/test_result.csv"
    )
    parser.parse()
    parser.get_accuracy()
