import numpy as np
import json
from memory_structures import Page, EpisodicSession, LinearMemory, GraphMemory, HierarchicalMemory
from qwen_client import get_client

class BestStructureEvaluator:
    lambda_q = 0.7
    lambda_m = 0.3

    def __init__(self):
        self._qwen = get_client()
        pass

    def find_best_structure(self, ep: list[Page], persona1: list[str], persona2: list[str]):
        rewards = np.zeros(3)
        structures = ["linear", "graph", "hierarchical"]
        for i in range(3):
            questions = self._generate_queries(persona1, persona2)
            session =  self._build_session_for_structure(ep, structures[i])
            responses = []
            for question in questions:
                query = question["query"]
                relevant_pages = self._retrieve_from_session(session, self._qwen.embed(query))
                prompt_for_response = self._build_response_prompt(query, relevant_pages)
                answer = self._qwen.generate(prompt_for_response)
                responses.append(answer)
            r_judge = self._compute_judge_reward(responses, questions)
            memory_layout_string = self._build_memory_layout_string(session)
            r_mem = self._compute_memory_util(memory_layout_string, session.summary)
            rewards[i] = self.lambda_q * r_judge + self.lambda_m * r_mem
        return np.argmax(rewards)
    
    def _generate_queries(self, persona1: list[str], persona2: list[str]) -> list:
        system_prompt_for_query_generation = """
            You are creating a test harness for an AI agent's memory.
            You will be given the ground-truth persona of 2 speakers (user and agent) in the following format:

            USER PERSONA:
            <list of facts>

            AGENT PERSONA:
            <list of facts>

            Your task is to generate 3 highly specific, natural questions that require remembering these exact details, along with their unambiguous, concise answers.

            Respond ONLY in this JSON format:
            {"questions": [{"query": "question string", "ground_truth_answer": "concise answer string"}, ...]}
            
            The questions attribute MUST have EXACTLY 3 questions.
            """
        user_prompt = """
            USER PERSONA:
            """ + "\n".join(persona1) + """

            AGENT PERSONA:
            """ + "\n".join(persona2) + """
        """
        response = self._qwen.chat(
            user=user_prompt,
            system=system_prompt_for_query_generation
        )
        try:
            obj = json.loads(response)
        except json.JSONDecodeError as e:
            print("Invalid JSON format in _generate_queries")
            raise
        if not obj["questions"] or len(obj["questions"]) != 3:
            print(obj)
            raise ValueError("Invalid questions object")

        return obj["questions"]
        
    def _build_response_prompt(
        self,
        query: str,
        mtem_pages: list[Page],
        user_a: str = "user_a",
        user_b: str = "agent",
    ) -> str:
      retrieval_lines = []
      for p in mtem_pages:
          retrieval_lines.append(f"- User: {p.user_text}")
          if p.agent_text:
              retrieval_lines.append(f"  Agent: {p.agent_text}")
      retrieval_text = "\n".join(retrieval_lines) if retrieval_lines else "(none)"
      prompt = (
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {user_a} or {user_b} "
        f"in an extremely concise manner.\n"
      )
      return prompt

    def _compute_judge_reward(self, responses: list[str], questions: list[str]) -> float:
        system_prompt_for_judgement = """
        You are an expert evaluator assessing an agent's memory retrieval success.
        You will be given the Ground-truth fact and the Predicted Answer.
        Analyze the Predicted Answer against the Ground Truth Fact. 

        Rate the factual correctness of ONLY the Predicted Answer on a scale from 0.0 (completely wrong/hallucinated) to 1.0 (perfectly accurate representation of the fact). 
        Do not penalize for minor syntactic variations, phrasing adjustments, or synonyms as long as the core fact matches.

        Your output MUST be a SINGLE float value between 0.0 and 1.0. Output NOTHING else.
        """

        total_score=0.0
        user_prompt = ""
        for i in range(len(questions)):
            user_prompt += f"""
                            Ground Truth Fact: "{questions[i]["ground_truth_answer"]}"
                            Predicted Answer: "{responses[i]}"

                            SCORE:
                            """
            response = self._qwen.chat(
                user=user_prompt,
                system=system_prompt_for_judgement
            )
            total_score += float(response)
        return total_score/5.0

    
    def _compute_memory_util(self, memory_layout_string, ground_truth_string):
        accuracy = self._simple_token_intersection(memory_layout_string, ground_truth_string)
        compression = 1 - (len(memory_layout_string.split()) / len(ground_truth_string.split()))
        return (0.6*accuracy + 0.4*compression)/2


    def _simple_token_intersection(self, memory_layout_string, ground_truth_string):
        import re
        def extract_meaningful_tokens(text):
            # Extract all alphanumeric words
            all_words = re.findall(r'\w+', text)
            meaningful_tokens = set()
            
            for word in all_words:
                if len(word) > 5:
                    meaningful_tokens.add(word.lower())
                elif any(char.isdigit() for char in word) or (word.isupper() and len(word) >= 3):
                    meaningful_tokens.add(word.lower())
                #Long words/proper nouns
                    
            return meaningful_tokens
        memory_set = extract_meaningful_tokens(memory_layout_string)
        gt_set = extract_meaningful_tokens(ground_truth_string)
        if not gt_set:
            return 0.0
        intersection = gt_set.intersection(memory_set)
        return len(intersection) / len(gt_set)

    def _build_session_for_structure(
    self,
    pages: list[Page],
    structure: str
) -> "EpisodicSession":
      """
      Build a *single* EpisodicSession containing all ``pages`` organised
      under ``structure`` with its index fully constructed.

      This is the correct unit of comparison: the same set of pages indexed
      three different ways, so retrieval differences are purely structural.
      """
      session = EpisodicSession(pages=list(pages), structure_type=structure)
      texts   = [p.to_text() for p in pages]
      summary = " ".join(texts)[:300]
      session.summary               = summary
      session.summary_embedding     = self._qwen.embed(summary)
      session.interaction_intensity = min(1.0, len(pages) * 0.1)

    # build the structure-specific index with ALL pages present
      if structure == "graph":
          GraphMemory().build_index(session)
      elif structure == "hierarchical":
          HierarchicalMemory().build_index(session)
      # linear needs no index

      return session

    def _retrieve_from_session(
    self,
    session: "EpisodicSession",
    query_emb: "np.ndarray",
    top_k: int = 3,
  ) -> list[Page]:
      """Retrieve pages from a session using its assigned structure."""
      if session.structure_type == "graph":
          return GraphMemory().retrieve(session, query_emb, top_k=top_k)
      elif session.structure_type == "hierarchical":
          return HierarchicalMemory().retrieve(session, query_emb, top_k=top_k)
      else:
          return LinearMemory().retrieve(session, query_emb, top_k=top_k)

    def _build_memory_layout_string(self, session: "EpisodicSession"):
      stored_memory = ""
      pages = session.pages
      if session.structure_type == "graph":
          id_to_num = {p.page_id: i + 1 for i, p in enumerate(pages)}
          adj = session.graph_index.get("adj", {}) if session.graph_index else {}
          for i, page in enumerate(session.pages):
              neighbour_ids = adj.get(page.page_id, [])
              neighbour_nums = sorted(id_to_num[nid] for nid in neighbour_ids if nid in id_to_num)
              neighbours_str = ",".join(str(n) for n in neighbour_nums) if neighbour_nums else "(none)"
              stored_memory += f"[Page {i+1} -> {neighbours_str}] {page.to_text()}"
      elif session.structure_type == "hierarchical":
          id_to_num = {p.page_id: i + 1 for i, p in enumerate(pages)}
          clusters = session.topic_tree.get("clusters", []) if session.topic_tree else []
          assigned_ids: set[str] = set()
          for c_idx, cluster in enumerate(clusters):
              cluster_num = c_idx + 1
              member_ids = cluster.get("members", [])
              # preserve original page order within the cluster
              member_nums = sorted((id_to_num[mid], mid) for mid in member_ids if mid in id_to_num)
              for page_num, pid in member_nums:
                  assigned_ids.add(pid)
                  page = pages[page_num - 1]
                  stored_memory += f"[Cluster {cluster_num}] [Page {page_num}] {page.to_text()}"
    
          # any pages not in a cluster (no embedding, or index not built)
          for i, page in enumerate(pages):
              if page.page_id not in assigned_ids:
                  stored_memory += f"[Cluster -] [Page {i + 1}] {page.to_text()}"
      else:
          for p in pages:
              stored_memory += f"[{p.timestamp}] {p.to_text()}\n"
      return stored_memory
