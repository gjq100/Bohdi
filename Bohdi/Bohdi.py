from Bohdi_Tree import *
from vllm import LLM, SamplingParams
from typing import Dict, List, Set, Tuple, Optional, Union
import re
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteriaList, StoppingCriteria, AutoModelForSequenceClassification
import argparse
import multiprocessing
import os
import torch
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import shutil
import yaml
from accelerate import Accelerator
import random
import ray
import time
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import gc
import math
import torch.multiprocessing as mp
import deepspeed

def detect_repetitive_data(text: str,
                           min_substring_len: int = 3,
                           max_substring_len: int = 100,
                           min_count: int = 3,
                           coverage_threshold: float = 0.6) -> bool:
    n = len(text)
    if n < min_substring_len * min_count:
        return False
    actual_max_substring_len = min(max_substring_len, n // 2)
    for length in range(min_substring_len, actual_max_substring_len + 1):
        substring_counts = {}
        for i in range(n - length + 1):
            substring = text[i : i + length]
            substring_counts[substring] = substring_counts.get(substring, 0) + 1
        max_count = 0
        if substring_counts:
             max_count = max(substring_counts.values())
        if max_count >= min_count:
            coverage_ratio = (max_count * length) / n
            if coverage_ratio >= coverage_threshold:
                return True
    return False

def ends_with_valid_ending(text: str) -> bool:
    if not text:
        return False
    
    if text.endswith('\n\n'):
        return True
    
    if text.endswith('. '):
        return True
    
    if text.endswith('.'):
        return True
    
    return False

@ray.remote(num_gpus=1, num_cpus=8)
class LLMActor:
    def __init__(
        self,
        model_name_or_path: str,
        max_model_len: int,
        device: Optional[Union[str, torch.device]] = None
    ):
        gpu_ids = ray.get_gpu_ids()
        # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        max_pos_emb = getattr(config, "max_position_embeddings", None)
        if max_pos_emb < max_model_len:
            max_model_len = max_pos_emb
        # Set the default CUDA device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        torch.cuda.set_device(0)
        # Initialize the LLM model
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=1,  
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.7,
            device = 'cuda:0'
        )

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _del(self):
        del self.llm.llm_engine.model_executor.driver_worker
        ray.shutdown()

    def generate(self, prompts, sampling_params, apply_temp = False):
        # Generate text using the LLM instance
        if apply_temp:
            prompts = [self._apply_chat_template(p) for p in prompts]
        outputs = self.llm.generate(prompts = prompts, sampling_params = sampling_params)
        return outputs


def formatting_prompts_func_single(example, tokenizer, eos_token):
    output_texts = []

    for i in range(len(example['instruction'])):
        text = [
            {"role": "user", "content": example['instruction'][i]}
        ]
        text = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True) + example['output'][i] + eos_token
        output_texts.append(text)

    return output_texts

class Bohdi:
    def __init__(self, 
                source_model_name_or_paths: List[str],
                target_model_name_or_path: str, 
                target_save_path: str,
                load_in_half: float,
                assigned_devices: int,
                temperatures: List[float],
                max_model_len: int,
                meditation_steps: int = 10,
                enlightenment_steps: int = 10,
                batch_size_meditation_phase: int = 10,
                batch_size_enlightenment_phase: int = 8,
                max_levels: int = 3,
                window_size: Union[int, str] = 'global',
                thr: float = 0.1,
                knowledge_tree_path: str = "knowledge_tree.json",
                basic_tree_save_path: str = "basic_tree.json",
                basic_tree_load_path: str = None):
        """
        Args:
            source_model_name_or_paths: List of paths of source models
            target_model_name_or_path: The path of the target model
            target_save_path: The path to save trained target model
            load_in_half: model load dtype
            assigned_devices: List of assigned devices
            temperatures: List of sampling temperatures
            max_model_len: Max input sequence length of each model
            meditation_steps: The number of sampling steps of meditation phase
            enlightenment_steps The number of steps for target model training
            batch_size_meditation_phase: The batch size for meditation phase
            batch_size_enlightenment_phase: The batch size for enlightenment phase
            target_model: The target model to be fine-tuned
            max_levels: Maximum depth of the tree
            window_size: The window size for SWBLRT
            thr: Threshold for SWBLRT
            knowledge_tree_path: Path to store the knowledge tree
            basic_tree_save_path: Path to store the whole basic tree
            basic_tree_load_path: Path to load the whole basic tree
        """
        self.tree = BasicTree(max_levels, knowledge_tree_path, basic_tree_save_path, basic_tree_load_path, window_size, thr)
        self.target_model_name_or_path = target_model_name_or_path
        self.source_model_name_or_paths = source_model_name_or_paths
        self.target_save_path = target_save_path
        self.load_in_half = load_in_half
        self.assigned_devices = assigned_devices
        self.temperatures = temperatures
        self.max_model_len = max_model_len
        self.window_size = window_size
        self.meditation_steps = meditation_steps
        self.enlightenment_steps = enlightenment_steps
        self.batch_size_meditation_phase = batch_size_meditation_phase
        self.batch_size_enlightenment_phase = batch_size_enlightenment_phase
    
    def _initialize_model(self, model_name_or_path: str, assigned_device: int, max_model_len: int, pg = None) -> LLM:
        """Initialize a single model with device placement"""
        model = LLMActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=assigned_device
                )
            ).remote(model_name_or_path, max_model_len, assigned_device)
        return model
    
    def _initialize_models(self, model_names_or_paths: List[str], assigned_devices: List[int], max_model_len: int, pg = None) -> List[LLM]:
        """Initialize multiple models with balanced device placement"""
        models = []
        for (idx, model_path) in enumerate(model_names_or_paths):
            models.append(self._initialize_model(model_path, assigned_devices[idx], max_model_len, pg = pg))
        return models
    
    def _cleanup_gpu_models(self):
        """Clean up models in GPUs"""
    
        device_count = torch.cuda.device_count()
        before = [torch.cuda.memory_allocated(i) for i in range(device_count)]
        
        if hasattr(self, 'target_model'):
            self.target_model._del.remote()
            del self.target_model
        
        if hasattr(self, 'source_models'):
            for model in self.source_models:
                model._del.remote()
                del model
            self.source_models.clear()

        gc.collect()
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            after = [torch.cuda.memory_allocated(i) for i in range(device_count)]
            for i in range(device_count):
                print(f'GPU {i}: Occupied {(after[i])/1024**2:.2f} MB')
    

    def _build_evaluation_prompt(self, qa_groups: List[Dict]) -> List[str]:
        """Build prompt for evaluating response quality"""
        
        response_template = """**Evaluation Criteria**\n
                            1. Accuracy (40%): Whether the answer correctly solves the problem and match the required response style.  
                            2. Clarity (30%): The expression is structured clearly and smoothly, and the content is concise without being verbose.
                            3. Completeness (20%): Whether all necessary details are covered.  
                            4. Relevance (10%): Whether the answer closely relates to the question. \n

                            **Additional Constraints**\n
                            Incoherent and repetitive answers should be directly assigned a low score, regardless of whether they contain the correct answer.

                            **Output Requirements**\n
                            1. You only need to reply with which answer you consider to be the best answer, and return the index (1-digit) of the best answer enclosed between [Best Answer Start] and [Best Answer End], following the format below: 
                            
                            The best answer is [Best Answer Start]index[Best Answer End].
                            
                            2. Only one index for the best answer can be returned."""

        evaluation_prompts = []
        for qa_group in qa_groups:
            question_str = f"""Please compare and evaluate the quality of the multiple answers to the following question, and return the index of the best one using **Arabic numerals**:\n
                        **Question** \n{qa_group['question']}\n
                        **Answers to be Evaluated**\n"""
            answers_str = "\n\n".join(
                        f"ANSWER {idx + 1}: \n{answer}" 
                        for (idx, answer) in enumerate(qa_group['answers'])
                        ) + '\n'
            evaluation_prompts.append(question_str + answers_str + response_template)

        return evaluation_prompts
    
    def meditation_phase(self):
        """
        Run one round of the meditation phase:
        1. Sample paths and generate questions
        2. Get responses from source and target models
        3. Evaluate responses using leader model
        4. Collect question-best answer pairs and related paths
        5. Update knowledge tree with rewards
        """

        ray.init(log_to_driver=True)
        pg = placement_group(
                name="llm_pg",
                bundles=[{"GPU": 1, "CPU": 8} for _ in range(len(self.assigned_devices))],
                strategy="STRICT_PACK" 
            )
        ray.get(pg.ready())

        print("Start Meditation")

        # Step 1: Sample questions and paths
        self.target_model = self._initialize_model(self.target_model_name_or_path, self.assigned_devices[0], self.max_model_len, pg = pg)
        self.source_models = self._initialize_models(self.source_model_name_or_paths, self.assigned_devices[1:], self.max_model_len, pg = pg)
        
        num_models = len(self.source_models) + 1
        
        rounds = 0
        while rounds < self.meditation_steps:
            print("Sampling Paths...")
            leader_idx = random.randint(0, int(len(self.source_models) - 1))
            questions, valid_paths, valid_useful_unks = self.tree.dynabranches(models = self.source_models, leader_idx = leader_idx, batch_size = self.batch_size_meditation_phase, for_min_opt = False, temperatures = self.temperatures[1:])
            
            if len(questions) > 0:
                
                #parallel processing
                responses = []
                generation_tasks = []

                illustration = """Answer the question below. Please add the “[Answer End]” marker at the end of your response once you believe you have completed the answer. \n\n Question: \n\n"""
                questions_for_generation = [illustration + q for q in questions]

                for (idx, model) in enumerate([self.target_model] + self.source_models):
                    generation_tasks.append(model.generate.remote(
                        prompts=questions_for_generation,
                        sampling_params=SamplingParams(
                            temperature=self.temperatures[idx],
                            top_p = 0.8,
                            n=1,
                            max_tokens=4096,
                            stop=["[Answer End]"]
                        ),
                        apply_temp = True))
                generation_results = ray.get(generation_tasks)

                for result in generation_results:
                    responses.append([result[i].outputs[0].text for i in range(len(questions))])
                
                qa_groups = []
                for idx, question in enumerate(questions):
                    valid_responses = [responses[i][idx] for i in range(num_models)]
                    qa_group = {'question': question, 'answers': valid_responses}
                    qa_groups.append(qa_group)

                # Step 3: Evaluate responses using leader model
                evaluation_prompts = self._build_evaluation_prompt(qa_groups)

                temperatures = self.temperatures[1:]
                print("Evaluating Responses...")
                evaluation_leader = self.source_models[leader_idx].generate.remote(
                                                        prompts=evaluation_prompts,
                                                        sampling_params=SamplingParams(
                                                            temperature=temperatures[leader_idx],
                                                            top_p=0.95,
                                                            max_tokens=4096,
                                                            n=5,  
                                                            stop=["[Best Answer End]"]
                                                        )
                                                    )
                evaluation_leader = ray.get(evaluation_leader)
                print("Collecting Best Answers...")

                # Step 4: Collect question-best answer pairs and related paths
                questions_with_best_answers = []
                valid_paths_with_valid_answer = []
                valid_useful_unks_with_valid_answer = []
                rewards = []
                for idx, evaluation_outputs in enumerate(evaluation_leader):
                    find_ans = False
                    for output in evaluation_outputs.outputs:
                        raw_text = output.text + "[Best Answer End]"
                        matches = re.findall(r'\[Best Answer Start\](\d+)\[Best Answer End\]', raw_text)
                        for match in matches:
                            if match.isdigit():
                                best_idx = int(match) - 1
                                if 0 <= best_idx < len(qa_groups[idx]['answers']):
                                    if (not detect_repetitive_data(qa_groups[idx]["answers"][best_idx])) and ends_with_valid_ending(qa_groups[idx]["answers"][best_idx]):
                                        if best_idx == 0:
                                            reward = 0
                                        else:
                                            reward = 1
                                        rewards.append(reward)
                                        questions_with_best_answers.append({"question": questions[idx], "answer": qa_groups[idx]["answers"][best_idx]})
                                        valid_paths_with_valid_answer.append(valid_paths[idx])
                                        valid_useful_unks_with_valid_answer.append(valid_useful_unks[idx])
                                        find_ans = True
                                        break
                        if find_ans == True:
                            break

                # Step 5: Update the tree params and the related JSON file
                print("Updating Tree Params...")
                self.tree._update_with_reward_feedback(valid_paths_with_valid_answer, valid_useful_unks_with_valid_answer, rewards, questions_with_best_answers)
                self.tree.save_to_file()
                rounds += 1
    

    def enlightenment_phase(self, current_loop=None, model_name_or_path=None):
        """Sample from exist paths and train the target model"""
        print("Start Minimization")
        # Initialize target model
    
        config = AutoConfig.from_pretrained(self.target_model_name_or_path)
        config.use_cache = False
    
        compute_dtype = (
                torch.bfloat16
                if self.load_in_half == "bf16"
                else (torch.float16 if self.load_in_half == "fp16" else torch.float32)
            )

        if 'gemma' in model_name_or_path.lower():
            config.cache_implementation = None
        
        model = transformers.AutoModelForCausalLM.from_pretrained(self.target_model_name_or_path,
                                                                config=config, torch_dtype=compute_dtype, device_map=None)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.target_model_name_or_path, padding=False, truncation=False, return_tensors="pt")
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token

        if 'llama' in model_name_or_path.lower():
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            lr = 5e-6
        elif 'qwen' in model_name_or_path.lower():
            response_template = "<|im_start|>assistant\n"
            lr = 5e-6
        elif 'gemma' in model_name_or_path.lower():
            response_template = "<start_of_turn>model\n"
            lr = 5e-6
        else:
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            lr = 5e-6
        collator_sft = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
        
        training_data = []
        for _ in range(self.enlightenment_steps):
            qa_pairs = self.tree.dynabranches(
                batch_size=self.batch_size_enlightenment_phase,
                for_min_opt=True
            )
            training_data.extend([{
                "instruction": qa["question"],
                "output": qa["answer"]
            } for qa in qa_pairs])

            
        training_args = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs = 1,
            learning_rate = lr,
            bf16 = True,
            optim = "adamw_torch_fused",
            max_seq_length = 8192,
            dataset_num_proc = 4,
            report_to = "none",
            output_dir = self.target_save_path,
            logging_steps=10,
            save_strategy="no",          
            save_steps=0,
            max_grad_norm=1.0,
            push_to_hub=False,
            packing = False,
            gradient_checkpointing=True
        )

        train_dataset = Dataset.from_list(training_data)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            formatting_func=lambda example: formatting_prompts_func_single(example, tokenizer, eos_token),
            data_collator=collator_sft,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        # Start training
        print(f"Starting SFT training with {len(training_data)} examples...")
    
        trainer.train()
        if torch.distributed.get_rank() == 0:
            final_path = os.path.join(self.target_save_path, "final_model")
            trainer.model.save_pretrained(final_path)
            trainer.tokenizer.save_pretrained(final_path)

            if current_loop is not None and current_loop % 10 == 0:
                periodic_path = os.path.join(self.target_save_path, f"periodic_checkpoint_loop_{current_loop}")
                trainer.model.save_pretrained(periodic_path)
                trainer.tokenizer.save_pretrained(periodic_path)
                print(f"\n[Periodic Save] Additional model saved at: {periodic_path}")
                
            checkpoint_dir = os.path.join(self.target_save_path, "checkpoint_new")
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.rename(final_path, checkpoint_dir)

            self.target_model_name_or_path = checkpoint_dir
            print(f"Training completed. Final model saved as checkpoint at: {checkpoint_dir}")
        torch.distributed.barrier()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and tokenize a dataset using a specified model.")
    parser.add_argument('--src_model_name_or_path', 
                    type=str, 
                    nargs='+',  
                    default=['/root/Qwen2.5-14B-Instruct'],
                    help="List of source model paths (space separated)")
    parser.add_argument('--tgt_model_name_or_path', type=str, default='/root/Llama-3.1-8B-Instruct', help="Target model path.")
    parser.add_argument('--tree_path', type=str, default='knowledge_tree.json', help="Tree save path.")
    parser.add_argument('--max_model_len', type=int, default=2048, help="Max Length of input sequence.")
    parser.add_argument('--target_save_path', type=str, default='./Ashvatta_Distilled', help="The save path of the trained target model.")
    parser.add_argument('--assigned_devices', type=int, 
                    nargs='+',  
                    default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="List of devices that the model are assigned.")
    parser.add_argument('--temperatures', 
                    type=float, 
                    nargs='+',  
                    default=[0.6, 0.7, 0.15, 0.7],
                    help="List of source model paths (space separated)")
    parser.add_argument("--load_in_half", type=str, default="bf16", help="none/fp16/bf16")
    parser.add_argument('--meditation_steps', type=int, default=2, help="Number of rounds of meditation phase.")
    parser.add_argument('--enlightenment_steps', type=int, default=5, help="Number of steps of enlightenment phase.")
    parser.add_argument('--batch_size_meditation_phase', type=int, default=200, help="Batch size of meditation phase.")
    parser.add_argument('--batch_size_enlightenment_phase', type=int, default=48, help="Batch size of enlightenment phase.")
    parser.add_argument('--basic_tree_save_path', type=str, default='basic_tree.json', help="Basic tree save path.")
    parser.add_argument('--basic_tree_load_path', type=str, default=None, help="Basic tree load path.")
    parser.add_argument('--step_now', type=int, default=None, help="Num of steps now.")
    parser.add_argument('--phase_id', type=str, default='enlightenment', help="The phase of Bohdi.")
    parser.add_argument('--window_size', type=str, default='global', help="The window size for SWBLRT.")
    parser.add_argument('--thr', type=float, default=0.1, help="Threshold for SWBLRT.")
    args = parser.parse_args()
    if args.window_size != 'global':
        window_size = int(args.window_size)
    else:
        window_size = args.window_size

    Bohdi_tree = Bohdi(source_model_name_or_paths = args.src_model_name_or_path,
                                target_model_name_or_path = args.tgt_model_name_or_path,
                                target_save_path = args.target_save_path,
                                load_in_half=args.load_in_half,
                                assigned_devices=args.assigned_devices,
                                temperatures=args.temperatures,
                                max_model_len = args.max_model_len,
                                meditation_steps = args.meditation_steps,
                                enlightenment_steps = args.enlightenment_steps,
                                batch_size_meditation_phase = args.batch_size_meditation_phase,
                                batch_size_enlightenment_phase = args.batch_size_enlightenment_phase,
                                max_levels = 3,
                                window_size = window_size,
                                thr = args.thr,
                                knowledge_tree_path = args.tree_path,
                                basic_tree_save_path = args.basic_tree_save_path,
                                basic_tree_load_path = args.basic_tree_load_path)
    
    print("Start to perform the " + str(args.step_now) + "-th loop.")
    if args.phase_id == 'meditation':
        if args.step_now > 0:
            if window_size != 'global':
                Bohdi_tree.tree.introspection_rebirth()
        Bohdi_tree.meditation_phase()
    else:
        Bohdi_tree.enlightenment_phase(current_loop = (args.step_now + 1), model_name_or_path = args.target_save_path)
    
