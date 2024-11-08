import os
import sys
import csv
import re
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


class TestAccuraciesParser:
    def __init__(self):
        self.files = {
            "prompt-001": [
                "../results/prompt-001/checkpoint-1362/test_result.csv",
                "../results/prompt-001/checkpoint-2043/test_result.csv",
                "../results/prompt-001/checkpoint-2724/test_result.csv",
                "../results/prompt-001/checkpoint-3405/test_result.csv",
                "../results/prompt-001/checkpoint-4086/test_result.csv",
                "../results/prompt-001/checkpoint-4767/test_result.csv",
                "../results/prompt-001/checkpoint-5448/test_result.csv",
                "../results/prompt-001/checkpoint-6129/test_result.csv",
                "../results/prompt-001/checkpoint-681/test_result.csv",
                "../results/prompt-001/checkpoint-6810/test_result.csv",
            ],
            # "prompt-002": [
            #     "../results/prompt-002/checkpoint-1120/test_result.csv",
            #     "../results/prompt-002/checkpoint-1260/test_result.csv",
            #     "../results/prompt-002/checkpoint-140/test_result.csv",
            #     "../results/prompt-002/checkpoint-1400/test_result.csv",
            #     "../results/prompt-002/checkpoint-280/test_result.csv",
            #     "../results/prompt-002/checkpoint-420/test_result.csv",
            #     "../results/prompt-002/checkpoint-560/test_result.csv",
            #     "../results/prompt-002/checkpoint-700/test_result.csv",
            #     "../results/prompt-002/checkpoint-840/test_result.csv",
            #     "../results/prompt-002/checkpoint-980/test_result.csv",
            # ],
            "prompt-003": [
                "../results/prompt-003/checkpoint-1239/test_result.csv",
                "../results/prompt-003/checkpoint-1652/test_result.csv",
                "../results/prompt-003/checkpoint-2065/test_result.csv",
                "../results/prompt-003/checkpoint-2478/test_result.csv",
                "../results/prompt-003/checkpoint-2891/test_result.csv",
                "../results/prompt-003/checkpoint-3304/test_result.csv",
                "../results/prompt-003/checkpoint-3717/test_result.csv",
                "../results/prompt-003/checkpoint-413/test_result.csv",
                "../results/prompt-003/checkpoint-4130/test_result.csv",
                "../results/prompt-003/checkpoint-826/test_result.csv",
            ],
            "prompt-004": [
                "../results/prompt-004/checkpoint-1320/test_result.csv",
                "../results/prompt-004/checkpoint-1760/test_result.csv",
                "../results/prompt-004/checkpoint-2200/test_result.csv",
                "../results/prompt-004/checkpoint-2640/test_result.csv",
                "../results/prompt-004/checkpoint-3080/test_result.csv",
                "../results/prompt-004/checkpoint-3520/test_result.csv",
                "../results/prompt-004/checkpoint-3960/test_result.csv",
                "../results/prompt-004/checkpoint-440/test_result.csv",
                "../results/prompt-004/checkpoint-4400/test_result.csv",
                "../results/prompt-004/checkpoint-880/test_result.csv",
            ],
            "prompt-005": [
                "../results/prompt-005/checkpoint-1208/test_result.csv",
                "../results/prompt-005/checkpoint-1510/test_result.csv",
                "../results/prompt-005/checkpoint-1812/test_result.csv",
                "../results/prompt-005/checkpoint-2114/test_result.csv",
                "../results/prompt-005/checkpoint-2416/test_result.csv",
                "../results/prompt-005/checkpoint-2718/test_result.csv",
                "../results/prompt-005/checkpoint-302/test_result.csv",
                "../results/prompt-005/checkpoint-3020/test_result.csv",
                "../results/prompt-005/checkpoint-604/test_result.csv",
                "../results/prompt-005/checkpoint-906/test_result.csv",
            ],
            "prompt-008": [
                "../results/prompt-008/checkpoint-1128/test_result.csv",
                "../results/prompt-008/checkpoint-1692/test_result.csv",
                "../results/prompt-008/checkpoint-2256/test_result.csv",
                "../results/prompt-008/checkpoint-2820/test_result.csv",
                "../results/prompt-008/checkpoint-3384/test_result.csv",
                "../results/prompt-008/checkpoint-3948/test_result.csv",
                "../results/prompt-008/checkpoint-4512/test_result.csv",
                "../results/prompt-008/checkpoint-5076/test_result.csv",
                "../results/prompt-008/checkpoint-564/test_result.csv",
                "../results/prompt-008/checkpoint-5640/test_result.csv",
            ],
        }

    def sorter(self):
        for key in self.files:
            self.files.get(key).sort(key=self._extract_checkpoint_number)

    def _extract_checkpoint_number(self, filename):
        match = re.search(r"checkpoint-(\d+)", filename)
        return int(match.group(1)) if match else float("inf")

    def get_accuracy_from_file(self, file: str):
        number_of_trial = 0
        number_of_correct_trial = 0
        with open(file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                number_of_trial += 1
                if row["truth"] == row["prediction"]:
                    number_of_correct_trial += 1
        return float(number_of_correct_trial) / float(number_of_trial)

    def get_accuracies_from_files(self):
        accuracies = {}
        for prompt in self.files:
            accuracies[prompt] = []
            for file in self.files.get(prompt):
                accuracies[prompt].append(self.get_accuracy_from_file(file=file))
        return accuracies

    def make_nepochs_accuracies_graph(
        self,
        accuracies,
        graph_name: str = f"{os.path.dirname(__file__)}/../deliverables/nepochs_accuracies_graph.png",
    ):
        plt.title("# epochs vs accuracies")
        plt.xlabel("# epochs")
        plt.ylabel("Accuracy")
        plt.grid()
        for prompt in accuracies:
            plt.plot(
                range(1, len(accuracies[prompt]) + 1, 1),
                accuracies[prompt],
                label=prompt,
            )

        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.savefig(graph_name)

        logger.info(f"{graph_name} was generated")


if __name__ == "__main__":
    parser = TestAccuraciesParser()
    parser.sorter()
    accuracies = parser.get_accuracies_from_files()
    parser.make_nepochs_accuracies_graph(accuracies=accuracies)
