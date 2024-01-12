import hashlib
import os
import time


def get_file_hash(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()

def get_file_hash_with_mtime(filename):
    mtime = os.path.getmtime(filename)
    with open(filename, 'rb') as f:
        file_hash = hashlib.sha256(f"{mtime}".encode())  # Include modification time in hash
        while chunk := f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()

def get_number_of_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

def check_for_new_experiment(current_hash, matrix_file):
    new_hash = get_file_hash(matrix_file)
    if new_hash != current_hash:
        new_experiment_length = get_number_of_lines(matrix_file)
        return new_hash, new_experiment_length
    return None, None

def get_latest_mtime(directory):
    max_mtime = 0
    found_file = False

    for root, _, files in os.walk(directory):
        for f in files:
            filepath = os.path.join(root, f)
            file_mtime = os.path.getmtime(filepath)
            if file_mtime > max_mtime:
                max_mtime = file_mtime
                found_file = True

    return max_mtime if found_file else None

def get_earliest_mtime(directory):
    min_mtime = float('inf')
    found_file = False

    for root, _, files in os.walk(directory):
        for f in files:
            filepath = os.path.join(root, f)
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < min_mtime:
                min_mtime = file_mtime
                found_file = True

    return min_mtime if found_file else None

def check_experiment_status(exp_folder):
    started_runs = 0
    finished_runs = 0
    for run_id in os.listdir(exp_folder):
        run_folder = os.path.join(exp_folder, run_id)
        if not os.path.isdir(run_folder):
            continue

        files = os.listdir(run_folder)
        if 'parameters.env' in files:
            started_runs += 1
            if len(files) > 1:
                finished_runs += 1

    return started_runs, finished_runs

def monitor(repeat, interval_seconds = 1, matrix_file='run_matrix.txt', exp_folder='./experiments', debug=False):
    if debug:
        start_monitor = time.time()
    experiment_length = None
    experiment_start_time = None
    while True:
        new_experiment_length = None
        if not os.path.exists(exp_folder):
            print("No experiment in progress.", end='\r', flush=True)
        else:
            new_experiment_length = get_number_of_lines(matrix_file)
            experiment_start_time = get_earliest_mtime(exp_folder)
            experiment_length = new_experiment_length
            print(f"\nExperiment started  at {time.ctime(experiment_start_time)}", end='\r', flush=True)
            started_runs, finished_runs = check_experiment_status(exp_folder)
            experiment_end_time = get_latest_mtime(exp_folder)
            experiment_duration = experiment_end_time - experiment_start_time
            experiment_duration_till_now = time.time() - experiment_start_time
            if finished_runs == experiment_length:
                print(
                    f"\nExperiment finished at {time.ctime(experiment_end_time)}.\n"
                    f"Total {experiment_duration//60:02,.0f}:{experiment_duration%60:02,.0f}. "
                    f"{experiment_length} runs, {experiment_duration/experiment_length:.2f} seconds per run."
                    , end='\r', flush=True)
            else:
                logstr = f" ({experiment_duration_till_now//60:02,.0f}:{experiment_duration_till_now%60:02,.0f} ago)"\
                    + f"\nRuns: running={started_runs-finished_runs:3.0f}, "\
                    + f"finished={finished_runs:3.0f}, "\
                    + f"permutations={experiment_length:3.0f}"
                logstr += f", {experiment_duration/finished_runs:5.1f}s per run" if finished_runs > 0 else ""
                print(logstr)
        if debug:
            print(f"monitor took {time.time() - start_monitor:,.3f} seconds", end='\r', flush=True)
        if not repeat:
            break  # Exit the loop if not repeating
        time.sleep(interval_seconds)

if __name__ == '__main__':
    if len(os.sys.argv) > 1:
        monitor(repeat=True, interval_seconds=int(os.sys.argv[1]))
    else:
        monitor(repeat=False)
