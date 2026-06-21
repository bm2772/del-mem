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

from torch import nn

input_dim: int = 14
num_structures: int = 3

class StructureClassifier(nn.Module):
    def __init__(self):
        super(StructureClassifier, self).__init__()
        self.prediction_seq = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_structures)
            #no softmax since cross entropy internally applies softmax
        )

    def forward(self, x):
        return self.prediction_seq(x)

class StructureDataset(Dataset):
    features_to_standardize: list = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]
    means: list = [0.0] * len(features_to_standardize)
    stds: list = [1.0] * len(features_to_standardize)

    def __init__(self, csvfilepath):
        df = pd.read_csv(csvfilepath).astype(float)

        self.data = torch.tensor(df.iloc[:, 0:input_dim].values, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

        #Standardize the data
        for i in self.features_to_standardize:
            mean = self.data[:, i].mean()
            std = self.data[:, i].std()
            self.means.append(mean)
            self.stds.append(std)
            self.data[:, i] = (self.data[:, i] - mean) / (std + 1e-8)  # Add a small value to avoid division by zero
        print(f"Standardized features: {self.features_to_standardize}")
        print(f"Feature means: {self.means}")
        print(f"Feature stds: {self.stds}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
