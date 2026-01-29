import json
import os
import math
import random
import sys
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import numpy as np
from typing import List
from pydantic import BaseModel
from tqdm import tqdm

from utils import json_chat, replace_prompt, VoteResponse, normal_chat

# Task Response Model
class TaskResponse(BaseModel):
    rationale: str
    name: str
    description: str
    shared_information: List[str]
    hidden_information: List[str]
    possible_answers: List[str]
    correct_answer: str

# Configuration
CONFIG = {
    "expected_validated_tasks": 100,  # Number of validated tasks we want
    "max_tasks_to_generate": 200,    # Maximum tasks to generate before stopping
    "model_for_generation": "gpt-4.1",
    "model_for_simulation": "gpt-4.1",
    "num_rounds": 15,
    "num_duplications": 10,
    "extra": "",
    "percentage_special_agents": 1.0,
    "validation_thresholds": {
        "complete_min_accuracy": 0.8,
        "hidden_max_accuracy": 0.2
    }
}

class Agent:
    """Agent class from sim.py"""
    def __init__(self, name, system_prompt, model, is_special=False):
        self.name = name
        self.is_special = is_special
        self.model = model
        self.history = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

    def chat(self, message):
        self.history.append({"role": "user", "content": message})
        response = normal_chat(self.history, model=self.model)
        self.history.append({"role": "assistant", "content": str(response)})
        return response
    
    def vote(self, message, possible_answers):
        local_history = self.history.copy()
        local_history.append({"role": "user", "content": message})
        response = json_chat(local_history, VoteResponse, model=self.model)
        return response

def load_prompts(prompt_dir="prompts"):
    """Load prompt templates from files."""
    prompts = {}
    prompt_files = [
        "system_prompt.txt",
        "first_user_prompt.txt",
        "user_prompt.txt",
        "first_vote_prompt.txt",
        "vote.txt",
        "generate.txt"  # Add generation prompt
    ]
    
    for file_name in prompt_files:
        filepath = os.path.join(prompt_dir, file_name)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                prompts[file_name.replace('.txt', '')] = f.read()
        else:
            print(f"Warning: {filepath} not found")
    
    return prompts

def generate_task(prompts, task_number):
    """Generate a new hidden profile task using the model."""
    generation_prompt = prompts.get("generate", 
        "Generate a hidden profile task where a group must make a decision. " +
        "The task should have shared information that all participants see, " +
        "and hidden information distributed among participants. " +
        "When all information is combined, there should be a clear correct answer.")
    
    messages = [
        {"role": "system", "content": generation_prompt}
    ]
    
    try:
        task = json_chat(messages, TaskResponse, model=CONFIG["model_for_generation"])
        # Add an ID to the task
        task_dict = task.dict() if hasattr(task, 'dict') else task
        task_dict['id'] = task_number
        return task_dict
    except Exception as e:
        print(f"Error generating task {task_number}: {e}")
        return None

def run_initial_votes_only(task, is_complete, prompts):
    """Run only initial votes for validation purposes. ALIGNED WITH SIM.PY"""
    model = CONFIG["model_for_simulation"]
    extra_prompt = CONFIG["extra"]
    percentage_special_agents = CONFIG["percentage_special_agents"]
    
    # Extract task details
    scenario_name = task["name"]
    description = task["description"]
    shared_info = task["shared_information"]
    hidden_info = task["hidden_information"]
    possible_answers = task["possible_answers"]
    correct_answer = task["correct_answer"]
    
    num_agents = len(hidden_info)
    num_special_agents = math.ceil(num_agents * percentage_special_agents)
    
    # Randomly select special agents
    num_special_agents = min(num_special_agents, num_agents)
    special_indices = random.sample(range(num_agents), num_special_agents)
    
    # Create agents - FOLLOWING SIM.PY PATTERN EXACTLY
    agents = []
    # Randomize the order of the hidden information (like sim.py)
    random.shuffle(hidden_info)
    
    for i in range(num_agents):
        # Assign hidden information to each agent - ensure it's a string
        information_list = shared_info.copy()  # Create a copy to avoid modifying original
        if not is_complete:
            information_list.append(hidden_info[i])
        else:
            # If is_complete, concatenate all hidden information (SHUFFLED VERSION)
            information_list.extend(hidden_info)
        random.shuffle(information_list)
        information_string = ""
        for index, piece in enumerate(information_list):
            information_string += f"Fact {str(index+1)}: {piece}\n"
        
        agent_prompt = replace_prompt(
            prompts["system_prompt"], {
                "description": description,
                "information": information_string,
                "extra": extra_prompt if i in special_indices else ""
            }
        )
        agent = Agent(f"Person {i+1}", agent_prompt, model, is_special=(i in special_indices))
        agents.append(agent)
    
    # Collect initial votes only
    initial_votes = []
    for agent in agents:
        vote_prompt = replace_prompt(prompts["first_vote_prompt"], {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(possible_answers)
        })
        
        vote_response = agent.vote(vote_prompt, possible_answers)
        # Store vote info like sim.py (without pre-calculated correctness)
        initial_votes.append({
            "agent": agent.name,
            "vote": vote_response['vote'],
            "rationale": vote_response['rationale']
        })
    
    # Calculate accuracy like automated_parsing.py
    initial_accuracy = sum([1 for vote_data in initial_votes if vote_data["vote"] == correct_answer]) / len(initial_votes) if initial_votes else 0
    
    return initial_accuracy

def run_full_scenario(task, is_complete, prompts):
    """Run full scenario with all discussion rounds for validated tasks. ALIGNED WITH SIM.PY"""
    model = CONFIG["model_for_simulation"]
    num_rounds = CONFIG["num_rounds"]
    extra_prompt = CONFIG["extra"]
    percentage_special_agents = CONFIG["percentage_special_agents"]
    
    # Extract task details
    scenario_name = task["name"]
    description = task["description"]
    shared_info = task["shared_information"]
    hidden_info = task["hidden_information"]
    possible_answers = task["possible_answers"]
    correct_answer = task["correct_answer"]
    
    num_agents = len(hidden_info)
    num_special_agents = math.ceil(num_agents * percentage_special_agents)
    
    # Set up result structure
    run_data = {
        "scenario": scenario_name,
        "is_complete": is_complete,
        "initial_votes": [],
        "final_votes": []
    }
    
    # Randomly select special agents
    num_special_agents = min(num_special_agents, num_agents)
    special_indices = random.sample(range(num_agents), num_special_agents)
    
    # Create agents - FOLLOWING SIM.PY PATTERN EXACTLY
    agents = []
    # Randomize the order of the hidden information (like sim.py)
    random.shuffle(hidden_info)
    
    for i in range(num_agents):
        # Assign hidden information to each agent - ensure it's a string
        information_list = shared_info.copy()  # Create a copy to avoid modifying original
        if not is_complete:
            information_list.append(hidden_info[i])
        else:
            # If is_complete, concatenate all hidden information (SHUFFLED VERSION)
            information_list.extend(hidden_info)
        random.shuffle(information_list)
        information_string = ""
        for index, piece in enumerate(information_list):
            information_string += f"Fact {str(index+1)}: {piece}\n"
        
        agent_prompt = replace_prompt(
            prompts["system_prompt"], {
                "description": description,
                "information": information_string,
                "extra": extra_prompt if i in special_indices else ""
            }
        )
        agent = Agent(f"Person {i+1}", agent_prompt, model, is_special=(i in special_indices))
        agents.append(agent)
    
    # Collect initial votes before discussion - FOLLOWING SIM.PY PATTERN
    for agent in agents:
        vote_prompt = replace_prompt(prompts["first_vote_prompt"], {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(possible_answers)
        })
        
        vote_response = agent.vote(vote_prompt, possible_answers)
        # Store like sim.py (without pre-calculated correctness)
        run_data["initial_votes"].append({
            "agent": agent.name,
            "vote": vote_response['vote'],
            "rationale": vote_response['rationale']
        })
    
    # Run discussion rounds - FOLLOWING SIM.PY PATTERN
    for round_num in range(num_rounds):
        if round_num == 0:
            prev_messages = []
            for agent_idx, agent in enumerate(agents):
                # Find the initial vote for this agent
                initial_vote_data = run_data["initial_votes"][agent_idx]
                initial_vote_prefix = f"Your initial vote was: {initial_vote_data['vote']}\nYour initial rationale was: {initial_vote_data['rationale']}\n\n"
                
                if not prev_messages:
                    prompt = initial_vote_prefix + prompts["first_user_prompt"]
                else:
                    messages_str = "\n".join(prev_messages)
                    base_prompt = replace_prompt(prompts["user_prompt"], {
                        "messages": messages_str,
                        "extra": extra_prompt if agent.is_special else ""
                    })
                    prompt = initial_vote_prefix + base_prompt
                
                response = agent.chat(prompt)
                prev_messages.append(f"{agent.name}: {response}")
        else:
            for agent in agents:
                prev_messages = []
                current_idx = agents.index(agent)
                for i in range(1, len(agents)):
                    idx = (current_idx + i) % len(agents)
                    a = agents[idx]
                    prev_msg = a.history[-1]["content"]
                    prev_messages.append(f"{a.name}: {prev_msg}")
                
                messages_str = "\n".join(prev_messages)
                prompt = replace_prompt(prompts["user_prompt"], {
                    "messages": messages_str,
                    "extra": extra_prompt if agent.is_special else ""
                })
                response = agent.chat(prompt)
    
    # Collect final votes after discussion - FOLLOWING SIM.PY PATTERN
    # Get the last round's discussion for context
    final_messages = []
    for agent in agents:
        final_messages.append(f"{agent.name}: {agent.history[-1]['content']}")
    group_discussion = "\n".join(final_messages)
    
    for agent in agents:
        vote_prompt = replace_prompt(prompts["vote"], {
            "group_discussion": group_discussion,
            "possible_answers": ", ".join(possible_answers)
        })
        
        vote_response = agent.vote(vote_prompt, possible_answers)
        # Store like sim.py (without pre-calculated correctness)
        run_data["final_votes"].append({
            "agent": agent.name,
            "vote": vote_response['vote'],
            "rationale": vote_response['rationale']
        })
    
    # Calculate accuracies like automated_parsing.py
    initial_accuracy = sum([1 for vote_data in run_data["initial_votes"] if vote_data["vote"] == correct_answer]) / len(run_data["initial_votes"]) if run_data["initial_votes"] else 0
    final_accuracy = sum([1 for vote_data in run_data["final_votes"] if vote_data["vote"] == correct_answer]) / len(run_data["final_votes"]) if run_data["final_votes"] else 0
    
    return initial_accuracy, final_accuracy

def validate_task(task, prompts):
    """Validate a task by running initial votes only."""
    print(f"\nValidating task: {task['name']}")
    
    complete_accuracies = []
    hidden_accuracies = []
    
    num_duplications = CONFIG["num_duplications"]
    
    # Run complete condition first (like sim.py)
    print("  Running complete condition validation...")
    with ThreadPoolExecutor() as executor:
        complete_futures = []
        for _ in range(num_duplications):
            complete_futures.append(executor.submit(run_initial_votes_only, task, True, prompts))
        
        # Process complete condition results
        for future in tqdm(as_completed(complete_futures), total=len(complete_futures), desc="Complete validation"):
            try:
                accuracy = future.result()
                complete_accuracies.append(accuracy)
            except Exception as e:
                print(f"Error in complete validation: {e}")
                traceback.print_exc()
    
    # Run hidden condition second (like sim.py)
    print("  Running hidden condition validation...")
    with ThreadPoolExecutor() as executor:
        hidden_futures = []
        for _ in range(num_duplications):
            hidden_futures.append(executor.submit(run_initial_votes_only, task, False, prompts))
        
        # Process hidden condition results
        for future in tqdm(as_completed(hidden_futures), total=len(hidden_futures), desc="Hidden validation"):
            try:
                accuracy = future.result()
                hidden_accuracies.append(accuracy)
            except Exception as e:
                print(f"Error in hidden validation: {e}")
                traceback.print_exc()
    
    # Calculate means (convert numpy types to Python native types)
    complete_mean = float(np.mean(complete_accuracies)) if complete_accuracies else 0.0
    hidden_mean = float(np.mean(hidden_accuracies)) if hidden_accuracies else 0.0
    complete_std = float(np.std(complete_accuracies)) if complete_accuracies else 0.0
    hidden_std = float(np.std(hidden_accuracies)) if hidden_accuracies else 0.0
    
    # Check validation thresholds (ensure boolean is Python native type)
    is_valid = bool(
        complete_mean > CONFIG["validation_thresholds"]["complete_min_accuracy"] and
        hidden_mean < CONFIG["validation_thresholds"]["hidden_max_accuracy"]
    )
    
    validation_result = {
        "task_name": task["name"],
        "complete_accuracy_before": complete_mean,
        "hidden_accuracy_before": hidden_mean,
        "complete_std_before": complete_std,
        "hidden_std_before": hidden_std,
        "is_valid": is_valid,
        "num_runs": num_duplications
    }
    
    print(f"  Complete condition (before): {complete_mean:.3f} ± {complete_std:.3f}")
    print(f"  Hidden condition (before): {hidden_mean:.3f} ± {hidden_std:.3f}")
    print(f"  Valid: {'✓' if is_valid else '✗'}")
    
    return validation_result

def run_full_evaluation(task, prompts):
    """Run full evaluation with discussion for validated tasks."""
    print(f"\nRunning full evaluation for validated task: {task['name']}")
    
    complete_before = []
    complete_after = []
    hidden_before = []
    hidden_after = []
    
    num_duplications = CONFIG["num_duplications"]
    
    # Run complete condition first (like sim.py)
    print("  Running complete condition...")
    with ThreadPoolExecutor() as executor:
        complete_futures = []
        for _ in range(num_duplications):
            complete_futures.append(executor.submit(run_full_scenario, task, True, prompts))
        
        # Process complete condition results
        for future in tqdm(as_completed(complete_futures), total=len(complete_futures), desc="Complete condition"):
            try:
                initial_acc, final_acc = future.result()
                complete_before.append(initial_acc)
                complete_after.append(final_acc)
            except Exception as e:
                print(f"Error in complete condition: {e}")
                traceback.print_exc()
    
    # Run hidden condition second (like sim.py)
    print("  Running hidden condition...")
    with ThreadPoolExecutor() as executor:
        hidden_futures = []
        for _ in range(num_duplications):
            hidden_futures.append(executor.submit(run_full_scenario, task, False, prompts))
        
        # Process hidden condition results
        for future in tqdm(as_completed(hidden_futures), total=len(hidden_futures), desc="Hidden condition"):
            try:
                initial_acc, final_acc = future.result()
                hidden_before.append(initial_acc)
                hidden_after.append(final_acc)
            except Exception as e:
                print(f"Error in hidden condition: {e}")
                traceback.print_exc()
    
    # Calculate statistics (convert numpy types to Python native types)
    complete_before_mean = float(np.mean(complete_before)) if complete_before else 0.0
    complete_before_std = float(np.std(complete_before)) if complete_before else 0.0
    complete_after_mean = float(np.mean(complete_after)) if complete_after else 0.0
    complete_after_std = float(np.std(complete_after)) if complete_after else 0.0
    hidden_before_mean = float(np.mean(hidden_before)) if hidden_before else 0.0
    hidden_before_std = float(np.std(hidden_before)) if hidden_before else 0.0
    hidden_after_mean = float(np.mean(hidden_after)) if hidden_after else 0.0
    hidden_after_std = float(np.std(hidden_after)) if hidden_after else 0.0
    
    full_eval_result = {
        "task_name": task["name"],
        "complete_before_mean": complete_before_mean,
        "complete_before_std": complete_before_std,
        "complete_after_mean": complete_after_mean,
        "complete_after_std": complete_after_std,
        "hidden_before_mean": hidden_before_mean,
        "hidden_before_std": hidden_before_std,
        "hidden_after_mean": hidden_after_mean,
        "hidden_after_std": hidden_after_std,
        "complete_change": complete_after_mean - complete_before_mean,
        "hidden_change": hidden_after_mean - hidden_before_mean,
        "num_runs": num_duplications
    }
    
    print(f"  Complete: Before={full_eval_result['complete_before_mean']:.3f}±{full_eval_result['complete_before_std']:.3f}, "
          f"After={full_eval_result['complete_after_mean']:.3f}±{full_eval_result['complete_after_std']:.3f}, "
          f"Change={full_eval_result['complete_change']:+.3f}")
    print(f"  Hidden: Before={full_eval_result['hidden_before_mean']:.3f}±{full_eval_result['hidden_before_std']:.3f}, "
          f"After={full_eval_result['hidden_after_mean']:.3f}±{full_eval_result['hidden_after_std']:.3f}, "
          f"Change={full_eval_result['hidden_change']:+.3f}")
    
    return full_eval_result

def save_results_incremental(all_tasks, validated_tasks, validation_results, full_eval_results, task_number, is_final=False):
    """Save results incrementally after each task is processed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = "data/generated_tasks"
    os.makedirs(results_dir, exist_ok=True)
    
    # Use a consistent timestamp for the session (from first save)
    if not hasattr(save_results_incremental, 'session_timestamp'):
        save_results_incremental.session_timestamp = timestamp
    
    session_timestamp = save_results_incremental.session_timestamp
    
    # Save all generated tasks
    all_tasks_file = os.path.join(results_dir, f"all_generated_tasks_{session_timestamp}.json")
    with open(all_tasks_file, 'w') as f:
        json.dump(all_tasks, f, indent=2)
    
    # Save validated tasks (main output)
    validated_tasks_file = os.path.join(results_dir, f"validated_tasks_{session_timestamp}.json")
    with open(validated_tasks_file, 'w') as f:
        json.dump(validated_tasks, f, indent=2)
    
    # Save validation and full evaluation results
    results_file = os.path.join(results_dir, f"evaluation_results_{session_timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "config": CONFIG,
            "timestamp": session_timestamp,
            "validation_results": validation_results,
            "full_evaluation_results": full_eval_results,
            "summary": {
                "total_generated": len(all_tasks),
                "total_validated": len(validated_tasks),
                "validation_rate": float(len(validated_tasks) / len(all_tasks)) if all_tasks else 0.0,
                "current_task_number": task_number,
                "is_final": is_final
            }
        }, f, indent=2)
    
    # Save progress checkpoint
    checkpoint_file = os.path.join(results_dir, f"checkpoint_{session_timestamp}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "task_number": task_number,
            "all_tasks_count": len(all_tasks),
            "validated_tasks_count": len(validated_tasks),
            "timestamp": session_timestamp,
            "config": CONFIG
        }, f, indent=2)
    
    if is_final:
        return validated_tasks_file, all_tasks_file, results_file
    else:
        return checkpoint_file

def save_results(all_tasks, validated_tasks, validation_results, full_eval_results):
    """Save all results to files (legacy function for final save)."""
    return save_results_incremental(all_tasks, validated_tasks, validation_results, full_eval_results, 0, is_final=True)

def print_summary(all_tasks, validated_tasks, validation_results, full_eval_results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TASK GENERATION AND EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Target validated tasks: {CONFIG['expected_validated_tasks']}")
    print(f"  Maximum tasks to generate: {CONFIG['max_tasks_to_generate']}")
    print(f"  Model: {CONFIG['model_for_simulation']}")
    print(f"  Rounds: {CONFIG['num_rounds']}, Duplications: {CONFIG['num_duplications']}")
    print(f"  Validation thresholds:")
    print(f"    Complete accuracy > {CONFIG['validation_thresholds']['complete_min_accuracy']}")
    print(f"    Hidden accuracy < {CONFIG['validation_thresholds']['hidden_max_accuracy']}")
    
    print(f"\nResults:")
    print(f"  Total tasks generated: {len(all_tasks)}")
    print(f"  Total tasks validated: {len(validated_tasks)}")
    print(f"  Validation rate: {len(validated_tasks)/len(all_tasks)*100:.1f}%" if all_tasks else "N/A")
    
    print(f"\n{'='*80}")
    print("VALIDATED TASKS - FULL EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Task':<30} {'Condition':<10} {'Before':<12} {'After':<12} {'Change':<10}")
    print("-" * 80)
    
    for task in validated_tasks:
        result = next((r for r in full_eval_results if r["task_name"] == task["name"]), None)
        if result:
            # Print complete condition
            print(f"{task['name'][:29]:<30} {'Complete':<10} "
                  f"{result['complete_before_mean']:.3f}±{result['complete_before_std']:.3f} "
                  f"{result['complete_after_mean']:.3f}±{result['complete_after_std']:.3f} "
                  f"{result['complete_change']:+.3f}")
            # Print hidden condition
            print(f"{'':<30} {'Hidden':<10} "
                  f"{result['hidden_before_mean']:.3f}±{result['hidden_before_std']:.3f} "
                  f"{result['hidden_after_mean']:.3f}±{result['hidden_after_std']:.3f} "
                  f"{result['hidden_change']:+.3f}")
            print()
    
    # Overall averages for validated tasks
    if full_eval_results:
        print("-" * 80)
        avg_complete_before = float(np.mean([r['complete_before_mean'] for r in full_eval_results]))
        avg_complete_after = float(np.mean([r['complete_after_mean'] for r in full_eval_results]))
        avg_hidden_before = float(np.mean([r['hidden_before_mean'] for r in full_eval_results]))
        avg_hidden_after = float(np.mean([r['hidden_after_mean'] for r in full_eval_results]))
        
        print(f"{'OVERALL AVERAGE':<30} {'Complete':<10} "
              f"{avg_complete_before:.3f}        "
              f"{avg_complete_after:.3f}        "
              f"{avg_complete_after - avg_complete_before:+.3f}")
        print(f"{'':<30} {'Hidden':<10} "
              f"{avg_hidden_before:.3f}        "
              f"{avg_hidden_after:.3f}        "
              f"{avg_hidden_after - avg_hidden_before:+.3f}")
    
    print(f"\n{'='*80}")
    print("FAILED TASKS (Validation)")
    print(f"{'='*80}")
    failed_count = 0
    for result in validation_results:
        if not result["is_valid"]:
            failed_count += 1
            print(f"  {result['task_name']}:")
            print(f"    Complete (before): {result['complete_accuracy_before']:.3f}")
            print(f"    Hidden (before): {result['hidden_accuracy_before']:.3f}")
    
    if failed_count == 0:
        print("  None - All generated tasks passed validation!")

def main():
    """Main function to generate and validate tasks."""
    print("Starting Hidden Profile Task Generation")
    print(f"Target: {CONFIG['expected_validated_tasks']} validated tasks")
    print(f"Maximum attempts: {CONFIG['max_tasks_to_generate']} tasks")
    
    # Load prompts
    prompts = load_prompts()
    
    # Initialize storage
    all_tasks = []
    validated_tasks = []
    validation_results = []
    full_eval_results = []
    
    # Generation loop
    task_number = 1
    pbar = tqdm(total=CONFIG['expected_validated_tasks'], desc="Validated tasks", position=0)
    
    while (len(validated_tasks) < CONFIG['expected_validated_tasks'] and 
           task_number <= CONFIG['max_tasks_to_generate']):
        
        print(f"\n{'='*60}")
        print(f"Generating task #{task_number}...")
        
        # Generate a new task
        task = generate_task(prompts, task_number)
        
        if task:
            all_tasks.append(task)
            
            # Validate the task (initial votes only)
            validation_result = validate_task(task, prompts)
            validation_results.append(validation_result)
            
            if validation_result["is_valid"]:
                validated_tasks.append(task)
                pbar.update(1)
                print(f"✓ Task validated! ({len(validated_tasks)}/{CONFIG['expected_validated_tasks']})")
                
                # Run full evaluation with discussion for validated task
                full_eval_result = run_full_evaluation(task, prompts)
                full_eval_results.append(full_eval_result)
                
                # Save results after each validated task
                checkpoint_file = save_results_incremental(
                    all_tasks, validated_tasks, validation_results, full_eval_results, task_number
                )
                print(f"  Progress saved to: {checkpoint_file}")
            else:
                print(f"✗ Task failed validation")
                # Save results after each task (even failed ones) to preserve progress
                checkpoint_file = save_results_incremental(
                    all_tasks, validated_tasks, validation_results, full_eval_results, task_number
                )
                print(f"  Progress saved to: {checkpoint_file}")
        else:
            print(f"Failed to generate task #{task_number}")
            # Save results even if generation failed
            checkpoint_file = save_results_incremental(
                all_tasks, validated_tasks, validation_results, full_eval_results, task_number
            )
            print(f"  Progress saved to: {checkpoint_file}")
        
        task_number += 1
    
    pbar.close()
    
    # Save final results
    validated_file, all_file, results_file = save_results_incremental(
        all_tasks, validated_tasks, validation_results, full_eval_results, task_number - 1, is_final=True
    )
    
    # Print summary
    print_summary(all_tasks, validated_tasks, validation_results, full_eval_results)
    
    print(f"\nFiles saved:")
    print(f"  Validated tasks: {validated_file}")
    print(f"  All generated tasks: {all_file}")
    print(f"  Evaluation results: {results_file}")
    
    if len(validated_tasks) >= CONFIG['expected_validated_tasks']:
        print(f"\n✓ Successfully generated {len(validated_tasks)} validated tasks!")
    else:
        print(f"\n⚠ Only generated {len(validated_tasks)} validated tasks out of {CONFIG['expected_validated_tasks']} target")

if __name__ == "__main__":
    main()