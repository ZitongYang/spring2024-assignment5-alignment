from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
import os
import re

from cs336_alignment.utils.io_utils import jdump, jload
from cs336_alignment.utils.io_utils import LEETER_TO_INT

@dataclass
class MultipleChoiceMMLU:
    question: str
    subject: str
    options: List[str]
    answer: int
    attempt: Optional[int] = None
    full_generation: Optional[str] = None

    @property
    def prompt(self) -> str:
        result = (
            f"Answer the following multiple choice question about {self.subject}. Respond with a single "
            f"sentence of the form \"The correct answer is _\", filling the blank with the letter "
            f"corresponding to the correct answer (i.e., A, B, C or D).\n"
            f"Question: {self.question}\n"
            f"A. {self.options[0]}\n"
            f"B. {self.options[1]}\n"
            f"C. {self.options[2]}\n"
            f"D. {self.options[3]}\n"
            f"Answer:\n"
        )
        return result

    def __repr__(self) -> str:
        return (
            f"### {self.subject} ###\n"
            f"### Question: ###\n"
            f"{self.question}\n"
            f"### Options: ###\n"
            f"A. {self.options[0]}\n"
            f"B. {self.options[1]}\n"
            f"C. {self.options[2]}\n"  # Assuming you meant self.options[2] here
            f"D. {self.options[3]}\n"
            f"### Answer: ###\n"
            f"{self.answer}\n"
            f"### Attempt: ###\n"
            f"Choice: {self.attempt}\n"
            f"Full generation: {self.full_generation}\n"
        )


class MMLU:
    def _parse_df(self, df: pd.DataFrame, subject: str) -> List[MultipleChoiceMMLU]:
        result = []
        for _, row in df.iterrows():
            question = row['question']
            options = [row['optionA'], row['optionB'], row['optionC'], row['optionD']]
            answer = LEETER_TO_INT[row['answer']]
            result.append(MultipleChoiceMMLU(question, subject, options, answer))
        return result
    
    def _parse_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None)
        column_names = ['question', 'optionA', 'optionB', 'optionC', 'optionD', 'answer']
        df.columns = column_names
        return df


    def __init__(self, split = 'test'):
        self.questions = []
        dir_path = f'data/mmlu/{split}'
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                subject = file[:-len(f'_{split}.csv')]
                df = self._parse_csv(os.path.join(dir_path, file))
                self.questions.extend(self._parse_df(df, subject))
    
    def evaluate_accuracy(self):
        correct = 0
        total = len(self.questions)
        for question in self.questions:
            if question.attempt == question.answer:
                correct += 1
        return correct / total

    def save_attempts(self, path: str):
        questions = [asdict(question) for question in self.questions]
        jdump(questions, path)
    
    def load_attempts(self, path: str):
        questions = jload(path)
        for question in questions:
            for q in self.questions:
                if q.question == question['question']:
                    q.attempt = question['attempt']
                    q.full_generation = question['full_generation']
                    break

    @staticmethod
    def parse_mmlu_response(model_output: str):
        if 'The correct answer is' in model_output:
            result =  model_output.split('The correct answer is ')[1][0]
            if result in ['A', 'B', 'C', 'D']:
                return LEETER_TO_INT[result]
        return 5
    
class GSM8K:
    @staticmethod
    def parse_gsm8k_response(model_output: str):
        matches = re.findall(r'(\d+)', model_output)
        if matches:
            return matches[-1]
        return None

if __name__ == '__main__':
    mmlu = MMLU()
    import pdb; pdb.set_trace()