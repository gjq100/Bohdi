import json
import os
from threading import Lock
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Union
import uuid
import random
from collections import defaultdict
import torch
import numpy as np
from collections import OrderedDict
import re
from vllm import LLM, SamplingParams
import sys
import ray
from scipy.stats import chi2

def show_path(path):
    names = []
    for node in path[1:]:
        names.append(node.name)
    return('→'.join(names))

class BasicNode:
    """Representation of a domain node in the tree"""
    def __init__(self, name: str, level: int, parent: 'BasicNode' = None, window_size: Union[int, str] = 'global'):
        self.name = name
        self.level = level
        self.parent = parent
        self.num_questions = 0
        self.children = OrderedDict()  # dict of nodes
        self.num_childs = 0

        if self.name == 'unk':
          self.is_unk = True
        else:
          self.is_unk = False

        if self.name == 'Root':
          self.beta_param = None
        else:
          self.beta_param = (1, 1)
        
        if window_size != 'global' and name != 'unk':
            self.window_size = window_size
            self.window = []
        else:
            self.window_size = None
    

    def add_child(self, node):
        """Add a child node with transition probability"""
        self.children[node.name] = node
        node.parent = self
        self.num_childs += 1
    
    def check_window(self):
        if len(self.window) < self.window_size:
            pass
        else:
            self.window = self.window[-self.window_size:]
    
    def SWBLRT(self, thr):
        if len(self.window) < self.window_size:
            return False
        else:
            alpha, beta = self.beta_param
            p_null = (alpha - 1) / (alpha + beta - 2)
            n = len(self.window)
            S = np.sum(self.window)
            p_mle = S / n
            log_lik_null = S * np.log(p_null + 1e-7) + (n - S) * np.log(1 - p_null + 1e-7)         
            log_lik_alt = S * np.log(p_mle + 1e-7) + (n - S) * np.log(1 - p_mle + 1e-7)    
            lambda_stat = 2 * (log_lik_alt - log_lik_null)
            p_value = 1 - chi2(df=1).cdf(lambda_stat)
            if p_value < thr:
                self.beta_param = (int(1 + S), int(1 + (len(self.window) - S)))
                return True
            else:
                return False
    
    def to_dict(self) -> dict:
        if self.window_size != None:
            return {
                "name": self.name,
                "level": self.level,
                "num_questions": self.num_questions,
                "window_size": self.window_size,
                "window": self.window,
                "beta_param": self.beta_param,
                "is_unk": self.is_unk,
                "children": [child.to_dict() for child in self.children.values()]  
            }
        else:
            return {
                "name": self.name,
                "level": self.level,
                "num_questions": self.num_questions,
                "window_size": None,
                "beta_param": self.beta_param,
                "is_unk": self.is_unk,
                "children": [child.to_dict() for child in self.children.values()] 
            }

    @classmethod
    def from_dict(cls, data: dict, parent: 'BasicNode' = None) -> 'BasicNode':
        node = cls(data["name"], data["level"], parent)
        node.num_questions = data["num_questions"]
        node.beta_param = tuple(data["beta_param"]) if data["beta_param"] else None
        node.is_unk = data["is_unk"]
        node.window_size = data["window_size"]
        if node.window_size is not None:
            node.window = data["window"]
        
        for child_data in data["children"]:
            child = cls.from_dict(child_data, parent=node)
            node.children[child.name] = child
            node.num_childs += 1
        
        return node


class BasicTree:
    def __init__(self, 
                max_levels: int = 3, 
                knowledge_tree_path: str = "knowledge_tree.json",
                basic_tree_save_path: str = "basic_tree.json",
                basic_tree_load_path: str = None,
                window_size: Union[int, str] = 'global',
                thr: float = 0.1):
        """
        Args:
            max_levels: Maximum depth of the tree (default 3: main→secondary→sub)
            knowledge_tree_path: The path of the JSON file related with the constructed knowledge tree
            basic_tree_save_path: The path to save the JSON file related with the constructed basic tree
            basic_tree_load_path: The path of the basic tree to load.
            window_size: Window size for SWBLRT Test
            thr: Threshold for Log-Ratio Test
        """
        self.max_levels = max_levels
        self.knowledge_tree_path = knowledge_tree_path
        self.basic_tree_save_path = basic_tree_save_path
        if window_size != 'global':
            self.window_size = window_size
        else:
            self.window_size = None
        self.thr = thr
        
        self.lock = Lock()
        if basic_tree_load_path is None:
            self._initialize_file()
            self.root = BasicNode("Root", level=0)
            self.root.add_child(BasicNode("unk", level=1))
            
            with self.lock:
                with open(self.knowledge_tree_path, 'r') as f:
                    dict_tree = json.load(f)     
                dict_tree["Root"] = {}
                temp_file = f"{self.knowledge_tree_path}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(dict_tree, f, indent=2)
                os.replace(temp_file, self.knowledge_tree_path)

            self.levels = {
                0: ['Root'],
                1: ['unk'],  # Main domains
                2: [],  # Secondary domains
                3: []   # Sub-domains
            }
            self.node_counter = {0: 1, 1: 1, 2: 1, 3: 1}  
        else:
            self.basic_tree_load_path = basic_tree_load_path
            self._load_from_file()
    
    def _initialize_file(self):
        """Initialize the knowledge tree structure as json"""
        with self.lock:
            if not os.path.exists(self.knowledge_tree_path):
                with open(self.knowledge_tree_path, 'w') as f:
                    json.dump({}, f)

    def save_to_file(self):
        
        data = {
            "max_levels": self.max_levels,
            "root": self.root.to_dict(), 
            "levels": self.levels,
            "node_counter": self.node_counter
        }
        with self.lock:
            temp_file = f"{self.basic_tree_save_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.basic_tree_save_path)

    def _load_from_file(self):
        with self.lock:
            with open(self.basic_tree_load_path, 'r') as f:
                data = json.load(f)
        data["levels"] = {int(k): v for k, v in data["levels"].items()}
        data["node_counter"] = {int(k): v for k, v in data["node_counter"].items()}
        self.max_levels = data["max_levels"]
        self.root = BasicNode.from_dict(data["root"]) 
        self.levels = data["levels"]
        self.node_counter = data["node_counter"]
        self._rebuild_parent_links(self.root)
    
    def _rebuild_parent_links(self, node: BasicNode, parent: BasicNode = None):
        node.parent = parent
        for child in node.children.values():
            self._rebuild_parent_links(child, parent=node)

    def get_all_domains(self):
        flat_list = []
        
        for level, domains in self.levels.items():
            valid = [d for d in domains]
            flat_list.extend(valid)
            
        return flat_list
    
    def _sample_path(self, models: List, except_unk: bool = False, temperatures: List[float] = None) -> List['BasicNode']:
        """Sample a path from root to leaf level"""
        path = [self.root]
        current_node = self.root
        useful_unks = []
        num_invalid_try = 0
        while (current_node.level < self.max_levels or current_node.name=='unk'):
            if current_node.name == 'unk':
                # expand a new domain at this level
                proposed_new_domain = self.sprout(path[:-1], models, temperatures)
                if proposed_new_domain is None:
                    # update beta distibution params
                    current_node.beta_param = (current_node.beta_param[0], current_node.beta_param[1] + 1)
                    # break the loop and re-sample at this level
                    current_node = path[-2]
                    path = path[:-1]
                    num_invalid_try += 1
                    if num_invalid_try == 10:
                        current_node.beta_param = (1, 1e7)
                        return 'invalid_node', None

                else:
                    current_node.beta_param = (current_node.beta_param[0] + 1, current_node.beta_param[1])
                    useful_unks.append(path[-1])
                    path[-1] = proposed_new_domain
                    current_node = proposed_new_domain
                    path[-2].add_child(current_node)
                    self.levels[current_node.level].append(current_node.name)
                    self.node_counter[current_node.level] += 1
                    # update the correlated tree in json file
                    with self.lock:
                        with open(self.knowledge_tree_path, 'r') as f:
                            dict_tree = json.load(f)    
                        if len(path) == 2:
                            new_node_name = path[1].name
                            dict_tree[path[0].name][new_node_name] = {}
                        elif len(path) == 3:
                            new_node_name = path[2].name
                            dict_tree[path[0].name][path[1].name][new_node_name] = {}
                        elif len(path) == 4:
                            new_node_name = path[3].name
                            dict_tree[path[0].name][path[1].name][path[2].name][new_node_name] = []

                        temp_file = f"{self.knowledge_tree_path}.tmp"
                        with open(temp_file, 'w') as f:
                            json.dump(dict_tree, f, indent=2)
                        os.replace(temp_file, self.knowledge_tree_path)
            
            if current_node.level < self.max_levels:
                if except_unk:
                    children = [child for key, child in current_node.children.items() if (key != 'unk' and child.num_questions > 0)]
                else:
                    children = list(current_node.children.values())
                
                alphas = np.array([c.beta_param[0] for c in children])
                betas = np.array([c.beta_param[1] for c in children])
            
                # Sampling on beta distribution to determine the next node 
                samples = np.random.beta(alphas, betas)
                next_node = children[np.argmax(samples)]
                
                path.append(next_node)
                current_node = next_node
        
        return path, useful_unks

    def sample_paths(self, batch_size: int, models: List, except_unk: bool = False, temperatures: List[float] = None) -> List[List['BasicNode']]:
        """Sample a batch of path from root to leaf level"""
        paths = []
        all_useful_unks = []
        for _ in range(batch_size):
            if except_unk:
                path, useful_unks = self._sample_path(models, except_unk = True)
            else:
                path, useful_unks = self._sample_path(models, temperatures = temperatures)
            if path != 'invalid_node':
                paths.append(path)
                all_useful_unks.append(useful_unks)
        return paths, all_useful_unks

    def sprout(self, path: List, 
                          source_models: List,
                          temperatures: List[float] = None) -> Optional['BasicNode']:
        """
        Handle domain expansion when encountering unknown node
        Args:
            path: The path to expand
            source_models: List of K source models for proposal generation
        Returns:
            New domain node if expansion succeeds, None otherwise
        """
        
        # Step 1: Generate proposals from source models
        prompt = self._build_expansion_prompt(path)
        proposals = []
        
        # parallel processing
        generation_tasks = []
        for idx, model in enumerate(source_models):
            generation_tasks.append(
                model.generate.remote(
                    prompts=[prompt],
                    sampling_params=SamplingParams(
                        temperature=temperatures[idx],
                        top_p=0.95,
                        max_tokens=2048,
                        n=5,
                        stop=["[Proposition End]"]
                    )
                )
            )

        generation_results = ray.get(generation_tasks)

        proposals = []
        for idx, outputs in enumerate(generation_results):
            for output in outputs[0].outputs:
                # Case-insensitive extraction with normalization
                raw_text = output.text + "[Proposition End]"
                if match := re.search(r'\[Proposition Start\](.+?)\[Proposition End\]', raw_text, re.DOTALL):
                    proposal = match.group(1).strip().title()  # Convert to Title Case
                    if self._validate_proposal(proposal):
                        proposals.append(proposal)
        
        # Step 2: Filter existing domains
        exist_domains = self.get_all_domains()
        unique_proposals = list(set(proposals) - set(exist_domains))
        
        # Step 3: Handle election cases
        if not unique_proposals:
            return None
            
        elif len(unique_proposals) == 1:
            new_domain = unique_proposals[0]
        else:
            counts = defaultdict(int)
            for p in proposals:
                if p in unique_proposals:
                    counts[p] += 1
            max_count = max(counts.values())
            candidates = [p for p, c in counts.items() if c == max_count]
            new_domain = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        print(new_domain)

        # Step 4: Add new domain to tree
        if self.window_size is not None:
            new_node = BasicNode(new_domain, level=path[-1].level + 1, window_size = self.window_size)
        else:
            new_node = BasicNode(new_domain, level=path[-1].level + 1)
        if new_node.level < 3:
            new_node.add_child(BasicNode('unk', level=new_node.level + 1))
        path[-1].add_child(new_node)
        return new_node

    def _build_expansion_prompt(self, path: List['BasicNode']) -> str:
        """Construct expansion prompt based on parent path"""

        parents = []
        path_in_str = show_path(path)
        for node in path:
            parents.append(node.name)
        
        level_headers = {
            '1': f"the domain you propose must be a secondary domain of {path[-1].name}.",
            '2': f"the domain you propose must be a specific sub-domain of {path[-1].name}."
        }

        headers = {
            '0': f'''I need to generate a hierarchical systematic knowledge tree. First, I need to determine a set of main subject domains, please use your world knowledge to propose a **Main Domain** that systematically taught in primary/secondary/higher education (e.g., in exact sciences, computer engineering, or other natural sciences and humanities), which should be as broad as possible to cover a wide range of child domains.''',
            '1/2': f'''This is a path of a hierarchical systematic knowledge tree: {path_in_str}, and now you need to propose a subject domain that logically and structurally follows this path, i.e., '''
            }
        
        constraints = f'''
                    **STRICT REQUIREMENTS**:
                    1. Must propose **EXACTLY ONE** new domain name
                    2. The proposed domain must be a clearly defined academic field related to **natural sciences** (such as physics, chemistry), **social sciences** (such as law, philosophy), **humanities** (linguistics, art), **formal sciences** (such as mathematics, computer science), or **interdisciplinary** fields (such as medicine, social psychology, etc.).
                    '''
        
        template = '''
                    **STRICT RESPONSE FORMAT**:
                    The proposed domain must be enclosed between [Proposition Start] and [Proposition End], following the format below:\n
                    [Proposition Start]proposed domain[Proposition End]\n
                    Now, please provide your proposed domain according to the requirements mentioned above.
                    '''
        
        if len(path) == 1:
            expansion_prompt = headers['0'] + '\n' + constraints + '\n' + template
        elif len(path) == 2:
            expansion_prompt = headers['1/2'] + level_headers['1'] + '\n' + constraints + '\n' + template
        elif len(path) == 3:
            expansion_prompt = headers['1/2'] + level_headers['2'] + '\n' + constraints + '\n' +  template
        else:
            raise ValueError("Out of Max Depth!")
        return expansion_prompt

    def _validate_proposal(self, proposal: str) -> bool:
        """Case-insensitive validation with Title Case normalization"""
        # Convert to Title Case first for validation
        normalized = proposal.title()
        
        return (
            re.fullmatch(r'^[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*){0,2}$', normalized) and
            normalized.lower() not in ['unk', 'unknown', 'none', 'and'] and "proposed" not in normalized.lower() and
            not any(c in normalized for c in '.,;:!?') and
            '  ' not in normalized  
        )
    
    def dynabranches(self, models: List = None, leader_idx: int = 0, batch_size: int = 10, for_min_opt: bool = False, temperatures: List[float] = None) -> Union[Tuple[List[str], List[List['BasicNode']]], List[Dict]]:
        """Sample a batch of paths and generate questions (when at meditation phase) or sample exist question and answers (when at enlightenment phase) for each path using the leader model.
        
        Args:
            models: List of source models
            leader_idx: The index of the leader model in the model list
            batch_size: Sample batch size
            for_min_opt: Whether the sampled batch of data is for the target model's training
            
        Returns:
            List of generated questions and related paths (when at meditation phase) or sampled questions and answers (when at enlightenment phase)
        """

        if for_min_opt:
            # for enlightenment phase, without generate new domains/questions, just sampling within exist qa pairs
            questions_and_answers = []
            sampled_paths, all_useful_unks = self.sample_paths(batch_size=batch_size, models=models, except_unk=True)
            with open(self.knowledge_tree_path, 'r') as f:
                dict_tree = json.load(f)     
            for path in sampled_paths:
                candidate_questions_and_answers = dict_tree[path[0].name][path[1].name][path[2].name][path[3].name]
                sampled_question_and_answer = random.choice(candidate_questions_and_answers)
                if len(sampled_question_and_answer["answer"]) > 30:
                    questions_and_answers.append(sampled_question_and_answer)
            return questions_and_answers

        else:
            # for meditation phase, sample questions with exploration of new domains/questions
            leader_model = models[leader_idx]  
            questions = []
            
            # Sample multiple paths (e.g., batch size of 10)
            sampled_paths, all_useful_unks = self.sample_paths(batch_size=batch_size, models=models, except_unk=False, temperatures=temperatures)
            
            question_prompts = []
            valid_paths = []
            valid_useful_unks = []
            for path in sampled_paths:
                # Build prompt for question generation based on the path    
                prompt = self._build_question_prompt(path)
                question_prompts.append(prompt)
            print("Generating Questions...")
            # Generate question using leader model
            batch_outputs = leader_model.generate.remote(
                prompts=question_prompts,
                sampling_params=SamplingParams(
                    temperature=temperatures[leader_idx],
                    top_p=0.95,
                    max_tokens=4096,
                    n=5,
                    stop=["[Question End]"]
                )
            )
            batch_outputs = ray.get(batch_outputs)
            
            # Harvest Operation
            for (idx, outputs) in enumerate(batch_outputs):
                # Extract the generated question
                for output in outputs.outputs:
                    raw_text = output.text + "[Question End]"
                    if match := re.search(r'\[Question Start\](.+?)\[Question End\]', raw_text):
                        question = match.group(1).strip()
                        if self._validate_question(question):
                            questions.append(question)
                            valid_paths.append(sampled_paths[idx])
                            valid_useful_unks.append(all_useful_unks[idx])
                            break
            return questions, valid_paths, valid_useful_unks
        
    def _build_question_prompt(self, path: List['BasicNode']) -> str:
        """Construct prompt for question generation based on the path"""
        domain_hierarchy = show_path(path)
        question_styles = [
            "The question should be a high-difficulty one that requires a step-by-step solution, with the answer numbered accordingly.",
            "The question should be open-ended and require the answer to include at least two different perspectives.",
            "The question should require coding to solve, with the answer presented in Markdown code block format.",
            "The question should require comparative analysis, with the answer displayed in a table format to show pros and cons.",
            "The question should require association with knowledge from other fields (e.g., math + music).",
            "The question should be styled as casual conversation and Q&A in daily life, with the tone and speaking style of the reply specified (e.g., using metaphors, rhyming)."
        ]
        selected_style = random.choice(question_styles)
        question_prompt = f"""Now I need to create high-quality SFT data for LLM training, so I need you to generate such data. \\
        For now, **you only need to create one question**. I will provide you with a specified main domain, its secondary domain, and a further refined sub-domain in the format [Main Domain]→[Secondary domain]→[Sub-Domain]. \\
        The corresponding topic is:
        
        {domain_hierarchy}
        
        The question must meet these requirements:
        1. Strictly fall within the scope of [{path[-1].name}] - neither too broad nor too narrow, and the stem of the question should first contain sufficient background information or relevant conditions
        2. The question you provide should be a relatively challenging, but it must be solvable, and the answer should be definitive
        3. {selected_style}
        4. Must be as original and concise as possible
        5. The expression style of the question should be **as diverse as possible**
        6. Enclose your response strictly between [Question Start] and [Question End] as shown below:

        [Question Start]Question[Question End]

        Now provide **EXACTLY ONE** question for the sub-domain **{path[3].name}** within secondary domain **{path[2].name}** of main domain **{path[1].name}**.
        """
        
        return question_prompt
    
    def _validate_question(self, question: str) -> bool:
        """Basic validation of generated questions"""
        return (
            len(question) > 30 and
            not any(invalid in question.lower() 
                   for invalid in ['sorry', "don't know"])
        )

    def introspection_rebirth(self):
        def _SWBLRT_recursive(node: BasicNode):
            if not node.is_unk and node.name != 'Root':
                result = node.SWBLRT(self.thr)
            for child in node.children.values():
                _SWBLRT_recursive(child)
        _SWBLRT_recursive(self.root)
    
    def check_all_windows(self):
        def _check_window_recursive(node: BasicNode):
            if not node.is_unk and node.name != 'Root':
                node.check_window()
            for child in node.children.values():
                _check_window_recursive(child)
        _check_window_recursive(self.root)

    def _update_with_reward_feedback(self, valid_paths: List[List['BasicNode']], valid_useful_unks: List[List['BasicNode']], rewards: List[float], questions_and_answers: Dict) -> str:
        """Update the dict in json and the beta params of each sampled paths."""
        with self.lock:
            with open(self.knowledge_tree_path, 'r') as f:
                dict_tree = json.load(f)   
            for (idx, valid_path) in enumerate(valid_paths):
                # Step-1: Update the beta distribution parameters of each node in path based on the reward feedback from the feasible paths.
                if rewards[idx] != 'None':
                    valid_path[1].beta_param = (valid_path[1].beta_param[0] + rewards[idx], valid_path[1].beta_param[1] + (1 - rewards[idx]))
                    valid_path[2].beta_param = (valid_path[2].beta_param[0] + rewards[idx], valid_path[2].beta_param[1] + (1 - rewards[idx]))
                    valid_path[3].beta_param = (valid_path[3].beta_param[0] + rewards[idx], valid_path[3].beta_param[1] + (1 - rewards[idx]))
                    if self.window_size is not None:
                        valid_path[1].window.append(int(rewards[idx]))
                        valid_path[2].window.append(int(rewards[idx]))
                        valid_path[3].window.append(int(rewards[idx]))

                valid_path[1].num_questions += 1
                valid_path[2].num_questions += 1
                valid_path[3].num_questions += 1
                #Step-2: Write the newly generated questions and the answers from each model (with the best answer at position 0) into the JSON dictionary.
                dict_tree[valid_path[0].name][valid_path[1].name][valid_path[2].name][valid_path[3].name].append(questions_and_answers[idx])
            
            if self.window_size is not None:
                self.check_all_windows()
            temp_file = f"{self.knowledge_tree_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(dict_tree, f, indent=2)
            os.replace(temp_file, self.knowledge_tree_path)
        
        
        
        