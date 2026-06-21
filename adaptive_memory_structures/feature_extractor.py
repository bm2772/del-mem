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

from __future__ import annotations
import json
import numpy as np
from memory_structures import (
    Page,
    _cosine,
)

# ---------------------------------------------------------------------------
# Conversation feature extraction (Appendix C, Table 5)
# ---------------------------------------------------------------------------

#NOTE: This function might require changes for fitting the features and their scales.
def extract_features(pages: list[Page]) -> np.ndarray:
    """
    Returns a 14-dim feature vector capturing structural conversation cues.
    All features are in numerical.
    """
    n = len(pages)
    if n == 0:
        return np.zeros(14, dtype=np.float32)

    page_count = n

    lengths = np.zeros(2*n)
    for i in range(n):
        lengths[2*i] = len(pages[i].user_text)
        lengths[2*i+1] = len(pages[i].agent_text)
    avg_len = np.mean(lengths)

    #Build prompt
    prompt_for_feature_extraction = """
    Here is a conversation between two speakers 1 and 2.

    Conversation:
    """ + "\n".join([f" Turn {2*i} : {p.user_text}\n Turn {2*i+1} : {p.agent_text}" for i, p in enumerate(pages)]) + """

    Extract the following information from the conversation:
    1. an array of the main topic of each dialogue turn IN ORDER
    2. all the unique entities mentioned (e.g., people, organizations, locations, events, activities, objects, concepts) along with their types and timestamps (if available) and any properties or attributes associated with these entities (e.g., "age", "location", "role", "appearance",etc.)
    3. relationships between these entities (e.g., "A works at B", "C lives in D") along with any temporal information about when these relationships were established or mentioned (if available)
    4. whether the conversation follows a question-answer pattern
    5. whether the conversation follows a decision tree pattern (e.g., successive branching based on choices)
    6. whether the conversation is entity-centric (i.e., revolves around a few key entities)

    Return ONLY valid JSON in this format:
    {
        "topics": ["topic_of_turn_0","topic_of_turn_1", ...],
        "entities": [{"id": "e1", "name": "Entity Name", "type": "PERSON|ORGANIZATION|LOCATION|EVENT|ACTIVITY|OBJECT|CONCEPT|OTHER", timestamp: "timestamp_if_available", properties: ["property1", "property2", ...]}, ...], 
        "relations": [{"source": "e1", "target": "e2", "relation": "lives in", timestamp: "since 2023-01-01"}, {"source": "e1", "target": "e3", "relation": "brother of", timestamp: "NONE"} ...],
        "is_qna_pattern": True/False,
        "is_decision_tree": True/False,
        "is_entity_centric": True/False
    }

    IMPORTANT: topics MUST HAVE EXACTLY """ + str(2*n) + """ elements
    Try to keep things as concise as possible."""


    from qwen_client import get_client
    qwen = get_client()
    response = None
    info = None
    try:
      response = qwen.chat(user=prompt_for_feature_extraction, system="You are a text analyzer. Your task is to analyze the conversation as mentioned and return a valid JSON.", max_new_tokens=1024)
      info = json.loads(response)
    except json.JSONDecodeError:
      print("Warning: Max tokens expanded")
      try:
        response = qwen.chat(user=prompt_for_feature_extraction, system="You are a text analyzer. Your task is to analyze the conversation as mentioned and return a valid JSON.", max_new_tokens=2048)
        info = json.loads(response)
      except json.JSONDecodeError:
        print("Warning: Info reduced")
        try:
          response = qwen.chat(user=prompt_for_feature_extraction, system="You are a text analyzer. Your task is to analyze the conversation as mentioned and return a valid JSON.\nIMPORTANT: Last response exceeded max tokens. Return only the important information.", max_new_tokens=2048)
          info = json.loads(response)
        except json.JSONDecodeError:
          print("Feature extraction aborted")
          return np.zeros(14, dtype=np.float32)


    entities = info.get("entities", [])
    entity_density = float(len(entities) / max(n,1))

    relations = info.get("relations", [])
    number_of_relations = len(relations)
    relation_indicators = float(number_of_relations / max(n, 1))

    topics = info.get("topics", [])
    topic_diversity = len(set(topics)) / max(n, 1)

    # 6. topic_transitions
    transitions = 0
    for i in range(1, len(topics)):
        if topics[i] != topics[i - 1]:
            transitions += 1
    topic_transitions = transitions / max(n - 1, 1)

    is_qna = float(info.get("is_qna_pattern", False))

    is_decision_tree = float(info.get("is_decision_tree", False))
    
    is_entity_centric = float(info.get("is_entity_centric", False))

    # 10. time_span (normalised to 1 hour)
    if n > 1:
        span = pages[-1].timestamp - pages[0].timestamp
        time_span = min(span / 3600.0, 1.0)
    else:
        time_span = 0.0

    # 11. temporal_density – number of timestamps / number of turns
    total_timestamp_count = 0
    timestamps = []
    for entity in entities:
        if "timestamp" in entity and entity["timestamp"]:
            total_timestamp_count += 1
            timestamps.append(entity['timestamp'])
    for relation in relations:
        if "timestamp" in relation and relation["timestamp"]:
            total_timestamp_count += 1
            timestamps.append(relation['timestamp'])

    temporal_density = float(total_timestamp_count / max(n, 1))
    
    # 12. semantic_complexity – mean pairwise distance between page embeddings
    if n > 1 and all(p.embedding is not None for p in pages):
        embs = np.stack([p.embedding for p in pages])
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_cosine(embs[i], embs[j]))
        semantic_complexity = 1.0 - float(np.mean(sims))
    else:
        semantic_complexity = 0.5

    lambda_e = 0.5
    lambda_t = 0.3
    lambda_c = 0.2

    turns = []
    for i in range(n):
        turns.append(pages[i].user_text.lower())
        turns.append(pages[i].agent_text.lower())
    turns_np = np.array(turns)

    rho_e = np.zeros(2*n)
    entity_names = [entity['name'] for entity in entities]
    for w in entity_names:
        rho_e += np.char.count(turns_np, w)
    total_fact_count = rho_e.sum() + number_of_relations
    rho_e = rho_e/lengths

    rho_t = np.zeros(2*n)
    for w in timestamps:
        rho_t += np.char.count(turns_np, w)
    rho_t = rho_t/lengths

    rho_c = np.zeros(2*n)
    discourse_markers = ["because", "therefore", "consequently", "as a result", "hence", "since", "however", "on the other hand", "whereas", "conversely",  "Although", "even though", "nevertheless", "despite", "in conclusion", "to sum up", "overall", "also", "in addition", "besides", "what's more", "In fact", "indeed", "clearly", "obviously"]
    for w in discourse_markers:
        rho_c += np.char.count(turns_np, w)
    rho_c = rho_c/lengths

    page_weights = lambda_e * rho_e + lambda_t * rho_t + lambda_c * rho_c
    page_weights = page_weights / np.sum(page_weights)
    avg_page_weight = np.mean(page_weights)
    H_norm = np.sum(- (page_weights * np.log(page_weights + 1e-4)))/(np.log(n) + 1e-4)
    above_avg_frac = np.count_nonzero(page_weights > avg_page_weight)/max(n,1)
    visual_salience_score = (1-H_norm)*above_avg_frac

    hyperedge_density = (total_fact_count) / max(n, 1)  #Facts per turn

    features = np.array([
        page_count, avg_len, entity_density, relation_indicators,
        topic_diversity, topic_transitions, is_qna, is_decision_tree,
        is_entity_centric, time_span, temporal_density, semantic_complexity,
        hyperedge_density, visual_salience_score
    ], dtype=np.float32)

    return features



# ---------------------------------------------------------------------------
# Original prompt for entity and relation extraction from conversation (not sure where to use it currently)
# ---------------------------------------------------------------------------
    """
    Extract entities and relationships from the following conversation.

                                    Conversation:
                                    <convo>

                                    Output format (JSON):
                                    {{"entities": [{"id": "e1", "name": "Entity Name", "type": "PERSON|ORGANIZATION|LOCATION|EVENT|TIME|ACTIVITY|OBJECT|CONCEPT|OTHER", "properties": {{}}}}, ...], "relations": [{{"source": "e1", "target": "e2", "relation": "relation_type", "weight": 1.0}}, ...]}}

                                    Important:
                                    - Extract ALL relevant entities including events, times, activities, objects, and concepts
                                    - For time entities, extract specific dates, months, years, or relative time expressions
                                    - For event entities, extract all mentioned events, meetings, activities
                                    - For activity entities, extract hobbies, interests, and regular activities
                                    - Extract relationships between entities when explicitly mentioned or clearly implied
                                    - Use concise names for entities
                                    - If no entities found, return empty arrays:
    """
