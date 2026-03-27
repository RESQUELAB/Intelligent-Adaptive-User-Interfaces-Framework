import psycopg2
import subprocess
from collections import defaultdict
import argparse
import csv
import concurrent.futures
import psutil
import time
import random
import os

def get_user_data(experiment_name=""):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST'),
            port=os.getenv('POSTGRES_PORT')
        )
        cursor = conn.cursor()

        # Get users in the experiment
        cursor.execute("""
            SELECT auth_user.id, human_feedback_api_profile.group 
            FROM auth_user
            INNER JOIN human_feedback_api_profile ON auth_user.id = human_feedback_api_profile.user_id
            WHERE auth_user.last_login IS NOT NULL AND human_feedback_api_profile.experiment = %s;
        """, (experiment_name,))
        user_group_map = {row[0]: row[1] for row in cursor.fetchall()}
        all_users = set(user_group_map.keys())

        # Get training completion records
        cursor.execute("""
            SELECT user_id, experiment, domain
            FROM human_feedback_api_trainingcompletion
            WHERE experiment = %s;
        """, (experiment_name,))
        
        completed_training = defaultdict(set)  # {user_id: {domain1, domain2, ...}}
        for user_id, _, domain in cursor.fetchall():
            completed_training[user_id].add(domain)

        cursor.close()
        conn.close()

        return all_users, completed_training, user_group_map
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return set(), {}, {}

def run_script(user_id, domains_to_train, experiment_name="test"):
    """Prepares training script commands for the user and domains."""
    commands = []
    for domain in domains_to_train:
        command = [
            "python", "rl_teacher/teach.py", "-e", "UIAdaptation-v0", "-n", experiment_name, "-p", "human", "-L", "10", 
              "-w", "1", "-tep", "100000", "-d", domain, "-c", "4", "-V", "-u", str(user_id), "-i", "1000000", "--force_new_reward_model", "--force_new_agent_model"        ]

            # "-w", "1", "-tep", "100000", "-d", domain, "-c", "4", "-V", "-u", str(user_id), "-i", "1000000"

        commands.append((user_id, domain, command))
    return commands

def execute_command(user_id, domain, command, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(command)}")
    else:
        try:
            wait_for_resources(max_cpu=80, max_memory=80)  # Wait before starting
            print(f"Executing command for user {user_id} with domain {domain}...")
            subprocess.run(command)
        except Exception as e:
            print(f"Error running command for user {user_id} with domain {domain}: {e}")

def wait_for_resources(max_cpu=90.0, max_memory=80.0, check_interval=1.0):
    """Waits until CPU and memory usage are below thresholds."""
    waited = False
    while True:
        # cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        if mem < max_memory:
            return waited
        waited = True
        print(f"[WAITING] Memory: {mem}%. Waiting for resources...")
        time.sleep(check_interval)


def get_incomplete_users(all_users, completed_training, all_domains):
    """ Returns users who did not complete training in all domains. """
    incomplete_users = []
    for user_id in all_users:
        completed_domains = completed_training.get(user_id, set())
        if completed_domains != all_domains:
            incomplete_users.append(user_id)
    return incomplete_users

def get_user_experiment_counts(completed_training):
    """ Returns how many times each (user, experiment) appears in the table. """
    user_experiment_counts = {user_id: len(domains) for user_id, domains in completed_training.items()}
    return user_experiment_counts

def export_to_csv(users, completed_training, filename="training_completion.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Username", "Completed Domains"])
        
        for user_id, username in users.items():
            domains = ", ".join(completed_training.get(user_id, []))
            writer.writerow([username, domains])
    print(f"CSV exported: {filename}")

GROUP_DOMAINS = {
    "1": {"courses"},
    "2": {"courses"},
    "3": {"trips"},
    "4": {"trips"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all users for an experiment.")
    parser.add_argument("-n", "--experiment_name", required=True, help="Name of the experiment")
    parser.add_argument("--dry_run", action="store_true", help="Only display information, do not train")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    dry_run = args.dry_run

    all_domains = {"trips", "courses"}  # Defined set of domains

    # Fetch user data
    all_users, completed_training, user_group_map = get_user_data(experiment_name)


    # Users who did not complete training in all domains
    incomplete_users = get_incomplete_users(all_users, completed_training, all_domains)
    print("Users who did not complete training:", incomplete_users)

    # How many times a user + experiment appears
    user_experiment_counts = get_user_experiment_counts(completed_training)
    print("User training counts:", user_experiment_counts)
    # for key in user_experiment_counts.keys():
    #     print(key, ": ",user_experiment_counts[key] )

    # Prepare list of commands to run
    all_commands = []
    print(user_group_map)
    for user_id, domains_completed in completed_training.items():
        if not domains_completed:
            print(f"[SKIP] User {user_id} – no completed domains.")
            continue
        all_commands.extend(run_script(user_id, domains_completed, experiment_name=experiment_name))

    # Limit number of concurrent training processes
    max_workers = 7
    dry_run_executed = 0 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (user_id, domain, command) in enumerate(all_commands):
            if not dry_run:
                waited_for_resources = wait_for_resources(max_cpu=80, max_memory=85)
            else:
                waited_for_resources = False

            futures.append(
                executor.submit(execute_command, user_id, domain, command, dry_run)
            )
            if dry_run:
                dry_run_executed += 1  #Count dry run command
                continue  # Don't delay in dry run

            if waited_for_resources:
                print(f"[PAUSE] Waited for resources, now sleeping 2 more minutes after launching task...")
                time.sleep(60)
            else:
                time.sleep(random.uniform(2, 3))
        if dry_run:
            print(f"[DRY RUN SUMMARY] Total commands that would have been executed: {dry_run_executed}")

        # Wait for all to finish and catch exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR in thread] {e}")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [
    #         executor.submit(execute_command, user_id, domain, command, dry_run)
    #         for user_id, domain, command in all_commands
    #     ]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()  # To raise any exceptions if they occurred

    # Train the agent for the completed domains
    # for user_id, domains_completed in completed_training.items():
    #     run_script(user_id, domains_completed, dry_run, experiment_name=experiment_name)
    # export_to_csv(all_users, completed_training)
