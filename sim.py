import json
import os
import math  # Add import for math.ceil
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import sys
import random
from tqdm import tqdm

from utils import json_chat, replace_prompt, VoteResponse, normal_chat

# Model name is now passed in as a command line argument
CONFIG_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"  # Default to gpt-4o if not specified

def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_benchmark(benchmark_path):
    """Load benchmark scenarios from file."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def load_prompts(prompt_dir="prompts"):
    """Load prompt templates from files."""
    prompts = {}
    prompt_files = [
        "system_prompt.txt",
        "first_user_prompt.txt",
        "user_prompt.txt",
        "first_vote_prompt.txt",
        "vote.txt"
    ]
    
    for file_name in prompt_files:
        with open(os.path.join(prompt_dir, file_name), 'r') as f:
            prompts[file_name.replace('.txt', '')] = f.read()
    
    return prompts

class Agent:
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

def run_scenario(benchmark_item, config_condition, prompts):
    """Run a single scenario with the given configuration."""
    model = config_condition["model"]
    # Determine number of agents based on hidden information
    hidden_info = benchmark_item["hidden_information"]
    num_agents = len(hidden_info)
    
    # Calculate number of special agents from percentage
    percentage_special_agents = config_condition["percentage_special_agents"]
    num_special_agents = math.ceil(num_agents * percentage_special_agents)
    
    num_rounds = config_condition["num_rounds"]
    extra_prompt = config_condition["extra"]
    is_complete = config_condition["is_complete"]
    
    # Extract scenario details
    scenario_name = benchmark_item["name"]
    description = benchmark_item["description"]
    shared_info = benchmark_item["shared_information"]
    possible_answers = benchmark_item["possible_answers"]
    
    # Set up the result structure
    run_data = {
        "scenario": scenario_name,
        "model": model,
        "num_agents": num_agents,
        "num_special": num_special_agents,
        "initial_votes": [],  # Add structure for initial votes before discussion
        "rounds": [],
        "final_votes": [],
        "majority_vote": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Randomly select agents to be special agents (ensure num_special_agents doesn't exceed num_agents)
    num_special_agents = min(num_special_agents, num_agents)
    special_indices = random.sample(range(num_agents), num_special_agents)
    
    # Create agents
    agents = []
    # Randomize the order of the hidden information
    random.shuffle(hidden_info)
    for i in range(num_agents):
        # Assign hidden information to each agent - ensure it's a string
        information_string = ""
        information_list = shared_info.copy()  # Create a copy to avoid modifying original
        if not is_complete:
            information_list.append(hidden_info[i])
        else:
            # If is_complete, concatenate all hidden information
            information_list.extend(hidden_info)
        random.shuffle(information_list)
        information_string = ""
        for index, piece in enumerate(information_list):
            information_string += f"Fact {str(index+1)}: {piece}\n"
        # Create agent with replaced prompts
        agent_prompt = replace_prompt(
            prompts["system_prompt"], {
                "description": description,
                "information": information_string,
                "extra": extra_prompt if i in special_indices else ""
            }
        )
        agent = Agent(f"Person {i+1}", agent_prompt, model, is_special=(i in special_indices))
        agents.append(agent)
    
    # Collect initial votes before any discussion
    for agent in agents:
        vote_prompt = replace_prompt(prompts["first_vote_prompt"], {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(possible_answers)
        })
        
        vote_response = agent.vote(vote_prompt, possible_answers)
        
        run_data["initial_votes"].append({
            "agent": agent.name,
            "vote": vote_response['vote'],
            "rationale": vote_response['rationale']
        })
    
    # Run discussion rounds
    for round_num in range(num_rounds):
        round_data = {
            "round": round_num + 1,
            "messages": [],
            "votes": []
        }
        
        # First round - agents speak in sequence
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
                
                round_data["messages"].append({
                    "agent": agent.name,
                    "prompt": prompt,
                    "response": response
                })

        # Subsequent rounds - each agent responds to previous messages
        else:
            for agent in agents:
                prev_messages = []
                # Get messages from previous round except current agent's
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
                
                round_data["messages"].append({
                    "agent": agent.name,
                    "prompt": prompt,
                    "response": response
                })
        
        # Only collect votes after the final round
        if round_num == num_rounds - 1:  # Final round
            group_discussion = "\n".join([f"{msg['agent']}: {msg['response']}" for msg in round_data["messages"]])
            
            # Use vote prompt for all post-discussion voting
            vote_prompt_template = prompts["vote"]
            
            for agent in agents:
                vote_prompt = replace_prompt(vote_prompt_template, {
                    "group_discussion": group_discussion,
                    "possible_answers": ", ".join(possible_answers)
                })
                
                vote_response = agent.vote(vote_prompt, possible_answers)
                
                round_data["votes"].append({
                    "agent": agent.name,
                    "vote": vote_response['vote'],
                    "rationale": vote_response['rationale']
                })
        # For non-final rounds, votes list remains empty (already initialized as [])
        
        run_data["rounds"].append(round_data)
    
    # Extract final votes from the last round
    run_data["final_votes"] = run_data["rounds"][-1]["votes"]
    
    # Calculate majority vote
    vote_counts = defaultdict(int)
    for vote_data in run_data["final_votes"]:
        vote_counts[vote_data["vote"]] += 1
    
    if vote_counts:
        majority_vote = max(vote_counts.items(), key=lambda x: x[1])[0]
        run_data["majority_vote"] = majority_vote
    
    return run_data

def save_intermediate_results(condition_results, results_store):
    """Save intermediate results after each scenario completion."""
    # Create tmp file path
    results_dir = os.path.dirname(results_store)
    results_filename = os.path.basename(results_store)
    name, ext = os.path.splitext(results_filename)
    tmp_file_path = os.path.join(results_dir, f"{name}_tmp_{condition_results['condition_id']}{ext}")
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the current condition results to tmp file
    with open(tmp_file_path, 'w') as f:
        json.dump(condition_results, f, indent=4)

def run_condition(benchmark_data, config_condition, prompts, results_store):
    """Run all scenarios for a single condition."""
    condition_id = config_condition["id"]
    num_duplications = config_condition["num_duplications"]
    model = config_condition["model"]
    percentage_special = config_condition["percentage_special_agents"]
    
    print(f"\nRunning condition {condition_id} with model {model} and {percentage_special*100:.0f}% special agents")
    
    # Create list of all scenarios to run
    scenarios = []
    for item in benchmark_data:
        for run in range(num_duplications):
            scenarios.append((run + 1, item))
    
    # Calculate total number of scenarios
    total_scenarios = len(scenarios)
    
    condition_results = {
        "condition_id": condition_id,
        "model": model,
        "percentage_special": percentage_special,
        "runs": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Run scenarios concurrently with progress bar
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_scenario = {
            executor.submit(run_scenario, item, config_condition, prompts): (run_num, item)
            for run_num, item in scenarios
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=total_scenarios, desc=f"Condition {condition_id} Progress", position=0) as pbar:
            for future in as_completed(future_to_scenario):
                run_num, item = future_to_scenario[future]
                try:
                    run_data = future.result()
                    condition_results["runs"].append(run_data)
                    
                    # Save intermediate results after each scenario completion
                    save_intermediate_results(condition_results, results_store)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "Scenario": item["name"],
                        "Run": run_num, 
                        "Agents": run_data["num_agents"],
                        "Special": run_data["num_special"]
                    })
                except Exception as e:
                    print(f"\nError in scenario {run_num} for {item['name']}: {str(e)}")
                    print("Error location:")
                    traceback.print_exc(file=sys.stdout)
    
    # Calculate summary statistics for this condition
    summary_stats = calculate_condition_stats(condition_results["runs"])
    condition_results["summary_statistics"] = summary_stats
    
    # Save final condition results to main file
    save_results(condition_results, results_store)
    
    return condition_results

def calculate_condition_stats(runs):
    """Calculate basic statistics for a condition's runs."""
    total_runs = len(runs)
    
    # Calculate average number of agents and special agents
    avg_num_agents = sum(run["num_agents"] for run in runs) / total_runs if total_runs > 0 else 0
    avg_num_special = sum(run["num_special"] for run in runs) / total_runs if total_runs > 0 else 0
    
    # Scenario-specific statistics
    scenario_stats = {}
    for run in runs:
        scenario = run["scenario"]
        if scenario not in scenario_stats:
            scenario_stats[scenario] = {
                "total_runs": 0,
                "num_agents": run["num_agents"]
            }
        
        scenario_stats[scenario]["total_runs"] += 1
    
    return {
        "total_runs": total_runs,
        "avg_num_agents": avg_num_agents,
        "avg_num_special": avg_num_special,
        "scenario_stats": scenario_stats
    }

def save_results(condition_results, results_store):
    """Save results to file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_store), exist_ok=True)
    
    # If file exists, append to it; otherwise, create new file
    if os.path.exists(results_store):
        with open(results_store, 'r') as f:
            all_results = json.load(f)
            
        # Check if condition results already exist and update if so
        for i, existing_condition in enumerate(all_results.get("conditions", [])):
            if existing_condition["condition_id"] == condition_results["condition_id"]:
                all_results["conditions"][i] = condition_results
                break
        else:
            # Condition doesn't exist yet, append it
            if "conditions" not in all_results:
                all_results["conditions"] = []
            all_results["conditions"].append(condition_results)
    else:
        # Create new results file
        all_results = {
            "conditions": [condition_results]
        }
    
    # Save updated results
    with open(results_store, 'w') as f:
        json.dump(all_results, f, indent=4)

def print_condition_summary(condition_id, stats):
    """Print summary statistics for a condition."""
    print(f"\nSummary Statistics for Condition {condition_id}:")
    print(f"Total runs: {stats['total_runs']}")
    print(f"Average agents per scenario: {stats['avg_num_agents']:.1f}")
    print(f"Average special agents per scenario: {stats['avg_num_special']:.1f}")
    
    print("\nScenario-specific statistics:")
    for scenario, scenario_stats in stats["scenario_stats"].items():
        print(f"\n{scenario} (Agents: {scenario_stats['num_agents']}):")
        print(f"  Total runs: {scenario_stats['total_runs']}")

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load configuration (from the script directory)
    config_path = os.path.join(script_dir, "configs", f"config_{CONFIG_MODEL}.json")
    config = load_config(config_path)
    
    # Load benchmark data (use paths from config)
    benchmark_path = os.path.join(script_dir, config["benchmark"])
    benchmark_data = load_benchmark(benchmark_path)
    
    # Load prompts (from the script directory)
    prompt_dir = os.path.join(script_dir, "prompts")
    prompts = load_prompts(prompt_dir)
    
    # Make sure store path is properly constructed
    store_path = os.path.join(script_dir, config["store"])
    
    # Run each condition
    all_results = {"conditions": []}
    for condition in config["conditions"]:
        if condition["run"]:
            condition_results = run_condition(benchmark_data, condition, prompts, store_path)
            all_results["conditions"].append(condition_results)
    
    print("\nAll conditions completed.")

if __name__ == "__main__":
    main()