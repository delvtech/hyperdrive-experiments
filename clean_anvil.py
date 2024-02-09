import subprocess
import sys
import numpy as np

GOOD_PROCESSES = 100 if not sys.argv[1] else int(sys.argv[1])


def get_anvil_cpu_usage():
    # Retrieves CPU usage and PID for processes with "anvil" in their command
    process_output = subprocess.check_output(["ps", "-eo", "%cpu,pid,cmd"]).decode()
    process_list = [line.split(None, 2) for line in process_output.split("\n")[1:] if "anvil" in line]

    return [(float(cpu), pid) for cpu, pid, _ in process_list]


def find_cutoff(cpu_usage, min_val, max_val, target_count):
    working_cutoffs = []
    too_low_cutoffs = []
    too_high_cutoffs = []
    for threshold in np.arange(min_val, max_val, 0.1):
        filtered_processes = [(cpu, pid) for cpu, pid in cpu_usage if cpu >= threshold]
        # print(f"Threshold: {threshold}, Processes: {len(cpu_usage)}, Above threshold: {len(filtered_processes)}")
        if len(filtered_processes) == target_count:
            working_cutoffs.append(threshold)
        elif len(filtered_processes) < target_count:
            too_high_cutoffs.append(threshold)
        elif len(filtered_processes) > target_count:
            too_low_cutoffs.append(threshold)
    if len(working_cutoffs) == 0 and len(too_low_cutoffs) > 0 and len(too_high_cutoffs) > 0:
        highest_too_low_cutoff = max(too_low_cutoffs)
        lowest_too_high_cutoff = min(too_high_cutoffs)
        for threshold in np.arange(highest_too_low_cutoff, lowest_too_high_cutoff, 0.01):
            filtered_processes = [(cpu, pid) for cpu, pid in cpu_usage if cpu >= threshold]
            # print(f"Threshold: {threshold}, Processes: {len(cpu_usage)}, Above threshold: {len(filtered_processes)}")
            if len(filtered_processes) == target_count:
                working_cutoffs.append(threshold)
    if len(working_cutoffs) == 0:
        return None
    return np.mean(working_cutoffs)


# Get anvil cpu usage
cpu_usage = get_anvil_cpu_usage()

# Find cutoff
cutoff = find_cutoff(cpu_usage, 3, 5, GOOD_PROCESSES)
if cutoff is None:
    print("Not terminating anvil processes.")
    exit(0)
processes_above = [(cpu, pid) for cpu, pid in cpu_usage if cpu >= cutoff]
processes_below = [(cpu, pid) for cpu, pid in cpu_usage if cpu < cutoff]
print(f"Cutoff: {cutoff}. Above: {len(processes_above)}, Below: {len(processes_below)}", end="")

if len(processes_below) == 0:
    print("")
    exit(0)

print(" (terminating)")

# Constructing the command for review
pids_to_terminate = " ".join(pid for _, pid in processes_below)
termination_command = f"kill -9 {pids_to_terminate}"
# print(f"Executing :\n{termination_command}")

# Execute the command
subprocess.call(termination_command, shell=True)
