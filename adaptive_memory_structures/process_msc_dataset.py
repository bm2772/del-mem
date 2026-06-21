'''
DATASET FORMAT:
12 FEATURES (as mentioned in FluxMem paper):
0. page_count 
1. avg_page_length
2. entity_density
3. relation_indicators
4. topic_diversity
5. topic_transitions
6. is_qna_pattern
7. is_decision_tree
8. is_entity centric
9. time_span
10. temporal_density
11. semantic_complexity

+2 extra features:
12. hyperedge_density
13. visual_salience_score

1 CLASS LABEL:
0 is hypergraph
1 is visual canvas
2 is vector store
'''

import numpy as np
import pandas as pd
from memory_structures import Page
from feature_extractor import extract_features
from best_structure_evaluator import BestStructureEvaluator
from datasets import load_dataset

from qwen_client import QwenClient
qwen = QwenClient.load("Qwen/Qwen3-4B")


class DatasetEvaluator:
    SECONDS_TO_READ_PER_WORD = 0.25
    SECONDS_TO_WRITE_PER_WORD = 1
    
    def __init__(self):
        self.structure_evaluator = BestStructureEvaluator()
        self.all_rows = []
        self.topic_hyperedges = {'dialoug_id': 0, 'content': ""}

    def eval_row(self, row):
        print(f"Dialogue_id: {row["dialoug_id"]}")
        if (self.topic_hyperedges["dialoug_id"] != row["dialoug_id"]):
                    self.topic_hyperedges["dialoug_id"] = row["dialoug_id"]
                    self.topic_hyperedges["content"] = ""

        ep = []
        n = len(row["dialogue"])
        timestamp=0
        i=0
        while i<n:
            user_text = ""
            if (row["speaker"][i] == 'Speaker 1'):
                user_text = row["dialogue"][i]
                i += 1
            agent_text = ""
            if (row["speaker"][i] == 'Speaker 2'):
                agent_text = row["dialogue"][i]
                i += 1
            timestamp += SECONDS_TO_WRITE_PER_WORD*len(user_text.split()) + SECONDS_TO_READ_PER_WORD*len(agent_text.split())
            ep.append(Page(user_text=user_text, agent_text=agent_text, timestamp=timestamp ,embedding=qwen.embed("USER: " + user_text + "\nAGENT: " + agent_text)))            

        features = extract_features(ep)
        if not np.any(features):
            return None
        label = self.structure_evaluator.find_best_structure(ep, row["persona1"], row["persona2"])
        res = {
            'page_count': features[0],
            'avg_page_length': features[1],
            'entity_density': features[2],
            'relation_indicators': features[3],
            'topic_diversity': features[4],
            'topic_transitions': features[5],
            'is_qna_pattern': features[6],
            'is_decision_tree': features[7],
            'is_entity centric': features[8],
            'time_span': features[9],
            'temporal_density': features[10],
            'semantic_complexity': features[11],
            'hyperedge_density': features[12],
            'visual_salience_score': features[13],
            'structure': label
        }
        self.all_rows.append(res)
    
    def save_to_csv(self, fileName: str):
        pd.DataFrame(self.all_rows).to_csv(fileName, index=False)


def main():
    dataset = load_dataset("nayohan/multi_session_chat")

    for w in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[w])
        df['dialoug_id'] = pd.to_numeric(df['dialoug_id'])
        df['session_id'] = pd.to_numeric(df['session_id'])
        df = df.sort_values(by=["dialoug_id", "session_id"]).reset_index(drop=True)
        dataset_evaluator = DatasetEvaluator()
        df.iloc[0:5].apply(dataset_evaluator.eval_row, axis=1)
        dataset_evaluator.save_to_csv(f"{w}_data.csv") 

main()
