import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
np.random.seed(42)
import copy
import time
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from lifted_pddl import Parser
import math
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

# class TupleRetrieverBM25:
#     def __init__(self, encode_type="bm25"):
#         self.documents = []  # Store reasoning text from tuples
#         self.tuples = []  # Store (reasoning, action, observation) tuples
#         assert encode_type in {"bm25", "tfidf", "embedding"}
#         self.encode_type = encode_type  #

#     def add_tuple(self, reasoning, action, observation):
#         """
#         Add a new (reasoning, action, observation) tuple to the retriever.
#         """
#         self.tuples.append((reasoning, action, observation))
#         self.documents.append(word_tokenize(reasoning))  # Tokenize reasoning text
#         self.bm25 = BM25Okapi(self.documents)  # Update BM25 index

#     def retrieve(self, query, k=3):
#         """
#         Retrieve the top-k most relevant tuples based on BM25 similarity to the query.
#         """
#         if not self.bm25:
#             return []  # No tuples to retrieve from
#         query_tokens = word_tokenize(query)
#         scores = self.bm25.get_scores(query_tokens)
#         top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
#         return [self.tuples[i] for i in top_indices]

class TupleRetriever:
    def __init__(self, retrieval_technique="bm25"):
        assert retrieval_technique in {"bm25", "tf-idf", "embedding"}, "Invalid retrieval technique specified"
        self.retrieval_technique = retrieval_technique
        self.corpus = ["Not Enough History"]
        self.tokenized_corpus = [["Not Enough History"]]
        self.index = None
        if retrieval_technique == "bm25":
            self.index = BM25Okapi(self.tokenized_corpus)
        elif retrieval_technique == "tf-idf":
            self.index = self.index = TfidfVectorizer()
        if self.retrieval_technique == "embedding":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_tuple(self, reasoning, action, observation):
        tuple_string = "\n".join([reasoning, action, observation])
        self.corpus.append(tuple_string)

        # Update the index dynamically
        if self.retrieval_technique == "bm25":
            tokenized_tuple = (tuple_string).split()
            self.tokenized_corpus.append(tokenized_tuple)
            self.index = BM25Okapi(self.tokenized_corpus)
        elif self.retrieval_technique == "tf-idf":
            self.index = TfidfVectorizer()
            self.index.fit(self.corpus)
        elif self.retrieval_technique == "embedding":
            self.embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def retrieve(self, query, top_k=3):
        if self.retrieval_technique == "bm25":
            query_tokens = query.split()
            scores = self.index.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [self.corpus[i] for i in top_indices]
        
        elif self.retrieval_technique == "tf-idf":
            query_vector = self.index.transform([query])
            corpus_vectors = self.index.transform(self.corpus)
            scores = (query_vector @ corpus_vectors.T).toarray().flatten()
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [self.corpus[i] for i in ranked_indices]
        
        elif self.retrieval_technique == "embedding":
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, self.embeddings).squeeze()
            ranked_indices = np.argsort(similarities.cpu().numpy())[::-1][:top_k]
            return [self.corpus[i] for i in ranked_indices]

class ResponseGenerator:
    def __init__(self, config_file, engine, verbose, ignore_existing):
        self.engine = engine
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.max_gpt_response_length = 500
        self.data = self.read_config(config_file)
        self.domain_pddl = f'./instances/{self.data["domain_file"]}'
        # self._set_task_params()
        if self.engine == "llama3.1":
            self.model = OllamaLLM(model="llama3.1")
        elif self.engine == 'bloom':
            self.model = self.get_bloom()
        elif 'finetuned' in self.engine:
            # print(self.engine)
            assert self.engine.split(':')[1] is not None
            model = ':'.join(self.engine.split(':')[1:])
            # print(model)
            self.engine='finetuned'
            self.model = {'model':model}
        else:
            self.model = None

    # def _set_task_params(self, instance_dir=None):
    #     if instance_dir is None:
    #         instance_dir = self.instance_dir
    #     else:
    #         self.instance_dir = instance_dir
    #     self.instance_folder = f'./instances/{instance_dir}/'
    #     self.instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
    #     self.n_files = min(self.data['n_instances'], len(os.listdir(self.instance_folder)))

    #     self.i_start = self.data['start']
    #     self.i_end = self.data['end']

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
        
    def get_bloom(self):
        max_memory_mapping = {0: "0GB", 1: "43GB", 2: "43GB", 3: "43GB", 4: "43GB", 5: "43GB"}
        cache_dir = os.getenv('BLOOM_CACHE_DIR', '/data/karthik/LLM_models/bloom/')
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir=cache_dir,
                                                     local_files_only=False, load_in_8bit=True, device_map='auto',
                                                     max_memory=max_memory_mapping)
        return {'model': model, 'tokenizer': tokenizer}

    def get_responses(self, task_name, specified_instances = [], run_till_completion=False):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
        os.makedirs(output_dir, exist_ok=True)
        output_json = output_dir+f"{task_name}.json"
        while True:
            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{task_name}.json")
                with open(prompt_dir+f"{task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine        
        
            failed_instances = []
            for instance in tqdm(structured_output["instances"]):
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        if self.verbose:
                            print(f"Instance {instance['instance_id']} already completed")
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])                   
                
                if self.verbose:
                    print(f"Sending query to LLM: Instance {instance['instance_id']}")
                query = instance["query"]
                stop_statement = "[STATEMENT]"
                if 'caesar' in self.data['domain_name']:
                    stop_statement = caesar_encode(stop_statement)
                llm_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model, stop=stop_statement)
                if not llm_response:
                    failed_instances.append(instance['instance_id'])
                    print(f"Failed instance: {instance['instance_id']}")
                    continue
                if self.verbose:
                    print(f"LLM response: {llm_response}")
                instance["llm_raw_response"] = llm_response
                with open(output_json, 'w') as file:
                    json.dump(structured_output, file, indent=4)
            
            if run_till_completion:
                if len(failed_instances) == 0:
                    break
                else:
                    print(f"Retrying failed instances: {failed_instances}")
                    time.sleep(5)
            else:
                break

    def check_goal_satisfaction(self, parser):
        current_state = parser.atoms
        goal_conditions = parser.goals
        return all(atom in current_state for atom in goal_conditions)
    
    def parse_reason_response(self, response):
        for line in response.splitlines():
            if "[END REASONING]" in line:
                return line
            
    def parse_action_response(self, response):
        for line in response.splitlines():
            if "[END ACTION]" in line:
                return line
            
    def regex_parse_action(self, response):
        action_pattern = r"Action:\s*(?:\(([^)]+)\)|([^()\[\]]+))\s*\[.*?\]"
        action_match = re.search(action_pattern, response)
        action_text = action_match.group(1) if action_match and action_match.group(1) else action_match.group(2)
        return action_text.strip() if action_text else None
    
    def regex_parse_reasoning(self, response):
        action_pattern = r"Reasoning:\s*(.*?)\s*\[END REASONING\]"
        action_match = re.search(action_pattern, response)
        action_text = action_match.group(1).strip() if action_match else None
        return action_text


    def get_react_responses(self, task_name, specified_instances = [], run_till_completion=False):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
        os.makedirs(output_dir, exist_ok=True)
        output_json = output_dir + f"{task_name}.json"

        if os.path.exists(output_json):
            with open(output_json, 'r') as file:
                structured_output = json.load(file)
        else:
            prompt_dir = f"prompts/{self.data['domain_name']}/"
            assert os.path.exists(prompt_dir + f"{task_name}.json")
            with open(prompt_dir + f"{task_name}.json", 'r') as file:
                structured_output = json.load(file)
            structured_output['engine'] = self.engine

        failed_instances = []
        count = -1
        for instance in tqdm(structured_output["instances"]):
            count += 1
            if count == 10:
                break
            if "llm_raw_response" in instance:
                if instance["llm_raw_response"] and not self.ignore_existing:
                    if self.verbose:
                        print(f"Instance {instance['instance_id']} already completed")
                    continue
            if len(specified_instances) > 0:
                if instance['instance_id'] not in specified_instances:
                    continue
                else:
                    specified_instances.remove(instance['instance_id'])

            if self.verbose:
                print(f"Starting ReAct loop for Instance {instance['instance_id']}")

            query = instance["query"]
            stop_statement = "[STATEMENT]"

            # lifted pddl parser
            parser = Parser()
            parser.parse_domain(self.domain_pddl)
            # cur_instance = f"./instances/{self.data['instance_dir']}/instance_{instance['instance_id']}.pddl"
            instance_dir = self.data['instance_dir']
            self.instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
            cur_instance = self.instance.format(instance["instance_id"])
            parser.parse_problem(cur_instance)
            # print(parser)
            gt_plan = instance["ground_truth_plan"]
            max_iterations = math.ceil(len(gt_plan.splitlines()) * 1.5)
            react_trace = []
            for step in range(max_iterations):
                # if self.verbose:
                #     print(f"Sending query: {query}")
                llm_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model)

                if not llm_response:
                    failed_instances.append(instance['instance_id'])
                    print(f"Failed instance: {instance['instance_id']}")
                    break

                reasoning, action = self.parse_reason_response(llm_response), self.parse_action_response(llm_response)
                reasoning = reasoning if reasoning is not None else ""
                action = action if action is not None else ""
                print("Reasoning:\n" + reasoning)
                print("Action:\n" + action)
                react_trace.append([reasoning, action])

                # react_pair = self.parse_response(llm_response)
                # react_pair = "" if react_pair is None else react_pair
                # print(step)
                # print(react_pair)
                # react_trace[step] = react_pair

                encoded_objects={
                'red': 'a',
                'blue': 'b',
                'orange': 'c',
                'yellow': 'd',
                'white': 'e',
                'magenta': 'f',
                'black': 'g',
                'cyan': 'h',
                'green': 'i',
                'violet': 'j',
                'silver': 'k',
                'gold': 'l'
                }

                try:
                    action_regex = self.regex_parse_action(action)
                    reasoning_regex = self.regex_parse_reasoning(reasoning)
                    print("action regex:\n" + action_regex)
                    print("reasoning regex:\n" + reasoning_regex)
                    print(1)
                    action_parts = action_regex.strip().split() 
                    print(2)
                    action_name = action_parts[0]
                    print(3)
                    action_parts = [encoded_objects[a] if a in encoded_objects else a for a in action_parts[1:]]
                    print(action_parts)
                    # action_args = tuple(ord(arg) - ord("a") for arg in action_parts[1:])
                    action_args = parser.get_object_indexes(action_parts)
                    print(4)
                    next_state = parser.get_next_state(action_name, action_args)
                    print(5)
                    observation = parser.encode_atoms_as_pddl(next_state, 'str')
                    print(6)
                    current_state = next_state
                    print(7)
                    print("Added query:\n")
                    print(f"Reasoning: {reasoning_regex} [END REASONING]\nAction: ({action_regex}) [END ACTION]\nObservation: {observation} [END OBSERVATION]\n\n")
                    query += f"Reasoning: {reasoning_regex} [END REASONING]\nAction: {action_regex} [END ACTION]\nObservation: {observation} [END OBSERVATION]\n\n"
                    if self.check_goal_satisfaction(parser):
                        query += "Goal achieved.\n"
                        break
                    print(8)
                except Exception as e:
                    print("APPLYING ACTION ERROR", e)
                    observation = f"Error applying action '{action}': {e}"
                    query += f"Observation: {observation}\n\n"
                    continue
            total_actions = "".join([chunk[1] for chunk in react_trace])
            print(total_actions)
            instance["llm_raw_response"] = total_actions
            instance["react_trace"] = react_trace
            with open(output_json, 'w') as file:
                json.dump(structured_output, file, indent=4)

        if run_till_completion and len(failed_instances) > 0:
            print(f"Retrying failed instances: {failed_instances}")
            time.sleep(5)
            self.get_react_responses(task_name, failed_instances, run_till_completion=True)

    def get_rag_react_responses(self, task_name, specified_instances = [], run_till_completion=False):
            output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
            os.makedirs(output_dir, exist_ok=True)
            output_json = output_dir + f"{task_name}.json"

            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir + f"{task_name}.json")
                with open(prompt_dir + f"{task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine

            failed_instances = []
            count = -1
            for instance in tqdm(structured_output["instances"]):
                count += 1
                if count == 10:
                    break
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        if self.verbose:
                            print(f"Instance {instance['instance_id']} already completed")
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])

                if self.verbose:
                    print(f"Starting ReAct loop for Instance {instance['instance_id']}")

                query = instance["query"]
                stop_statement = "[STATEMENT]"

                # lifted pddl parser
                parser = Parser()
                parser.parse_domain(self.domain_pddl)
                # cur_instance = f"./instances/{self.data['instance_dir']}/instance_{instance['instance_id']}.pddl"
                instance_dir = self.data['instance_dir']
                self.instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
                cur_instance = self.instance.format(instance["instance_id"])
                parser.parse_problem(cur_instance)
                gt_plan = instance["ground_truth_plan"]
                max_iterations = math.ceil(len(gt_plan.splitlines()) * 1.5)
                react_trace = []
                tuple_retriever = TupleRetriever(retrieval_technique="embedding")
                for step in range(max_iterations):
                    current_st = f"Current State: {parser.encode_atoms_as_pddl(parser.atoms, 'str')}"
                    retrieved_tuples = tuple_retriever.retrieve(current_st, top_k=3) if len(tuple_retriever.corpus) >= 3 else []
                    query += "\n\nReference History:\n\n"
                    for d in retrieved_tuples:
                        query += f"{d}\n\n"
                    query += f"\n\nEnd of Reference History\n\n{current_st}\n\nNow, generate the next Reasoning and Action step. MAKE SURE to follow to this format:\nReasoning: (insert reasoning) [END REASONING]\nAction: (insert action) [END ACTION]:\n"
                    if self.verbose:
                        print(f"Sending query: {query}")
                    llm_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model)

                    if not llm_response:
                        failed_instances.append(instance['instance_id'])
                        print(f"Failed instance: {instance['instance_id']}")
                        break
                    print(llm_response)
                    reasoning, action = self.parse_reason_response(llm_response), self.parse_action_response(llm_response)
                    reasoning = reasoning if reasoning is not None else ""
                    action = action if action is not None else ""
                    print("Reasoning:\n" + reasoning)
                    print("Action:\n" + action)
                    obs = parser.encode_atoms_as_pddl(parser.atoms, 'str')
                    print(type(reasoning), type(action), type(obs))
                    tuple_retriever.add_tuple(reasoning, action, "Observation: " + str(obs))
                    react_trace.append([reasoning, action, retrieved_tuples])

                    # react_pair = self.parse_response(llm_response)
                    # react_pair = "" if react_pair is None else react_pair
                    # print(step)
                    # print(react_pair)
                    # react_trace[step] = react_pair

                    encoded_objects={
                    'red': 'a',
                    'blue': 'b',
                    'orange': 'c',
                    'yellow': 'd',
                    'white': 'e',
                    'magenta': 'f',
                    'black': 'g',
                    'cyan': 'h',
                    'green': 'i',
                    'violet': 'j',
                    'silver': 'k',
                    'gold': 'l'
                    }
            

                    try:
                        action_regex = self.regex_parse_action(action)
                        reasoning_regex = self.regex_parse_reasoning(reasoning)
                        print("action regex:\n" + action_regex)
                        print("reasoning regex:\n" + reasoning_regex)
                        print(1)
                        action_parts = action_regex.strip().split() 
                        print(2)
                        action_name = action_parts[0]
                        print(3)
                        action_parts = [encoded_objects[a] if a in encoded_objects else a for a in action_parts[1:]]
                        print(action_parts)
                        # action_args = tuple(ord(arg) - ord("a") for arg in action_parts[1:])
                        action_args = parser.get_object_indexes(action_parts)
                        print(4)
                        next_state = parser.get_next_state(action_name, action_args)
                        print(5)
                        observation = parser.encode_atoms_as_pddl(next_state, 'str')
                        print(6)
                        current_state = next_state
                        print(7)
                        print("Added query:\n")
                        print(f"Reasoning: {reasoning_regex} [END REASONING]\nAction: ({action_regex}) [END ACTION]\nObservation: {observation} [END OBSERVATION]\n\n")
                        query += f"Reasoning: {reasoning_regex} [END REASONING]\nAction: {action_regex} [END ACTION]\nObservation: {observation} [END OBSERVATION]\n\n"
                        if self.check_goal_satisfaction(parser):
                            query += "Goal achieved.\n"
                            break
                        print(8)
                    except Exception as e:
                        print("APPLYING ACTION ERROR", e)
                        observation = f"Error applying action '{action}': {e}"
                        query += f"Observation: {observation}\n\n"
                        continue
                total_actions = "".join([chunk[1] for chunk in react_trace])
                print(total_actions)
                instance["llm_raw_response"] = total_actions
                instance["react_trace"] = react_trace
                with open(output_json, 'w') as file:
                    json.dump(structured_output, file, indent=4)

            if run_till_completion and len(failed_instances) > 0:
                print(f"Retrying failed instances: {failed_instances}")
                time.sleep(5)
                self.get_react_responses(task_name, failed_instances, run_till_completion=True)


if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t2 = Optimal Planning \
    \n t3 = Plan Verification \
    \n t4 = Plan Reuse\
    \n t5 = Plan Generalization\
    \n t6 = Replanning (easier) \
    \n t7 = Reasoning about Plan Execution \
    \n t8_1 = Goal Reformulation (Goal shuffling) \
    \n t8_2 = Goal Reformulation (Full -> Partial) \
    \n t8_3 = Goal Reformulation (Partial -> Full) \
    ')
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
                        
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    # parser.add_argument('--random_example', type=str, default="False", help='Random example')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    config = args.config
    specified_instances = args.specific_instances
    verbose = eval(args.verbose)
    run_till_completion = eval(args.run_till_completion)
    ignore_existing = args.ignore_existing
    print(f"Task: {task}, Engine: {engine}, Config: {config}, Verbose: {verbose}, Run till completion: {run_till_completion}")
    # specified_instances = args.specified_instances
    # random_example = eval(args.random_example)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing)
    task_dict = {
        't1': 'task_1_plan_generation',
        't2': 'task_2_plan_optimality',
        't3': 'task_3_plan_verification',
        't4': 'task_4_plan_reuse',
        't5': 'task_5_plan_generalization',
        't6': 'task_6_replanning',
        't7': 'task_7_plan_execution',
        't8_1': 'task_8_1_goal_shuffling',
        't8_2': 'task_8_2_full_to_partial',
        't8_3': 'task_8_3_partial_to_full',
    }
    try:
        task_name = task_dict[task]
    except:
        raise ValueError("Invalid task name")
    response_generator.get_responses(task_name, specified_instances, run_till_completion=run_till_completion)





