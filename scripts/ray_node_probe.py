import subprocess

import ray


def run(cmd, timeout=20):
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout
        )
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@ray.remote(num_cpus=0)
def probe(ip):
    return {
        "ip": ip,
        "hostname": run("hostname", 5).strip(),
        "date": run('date "+%F %T %Z"', 5).strip(),
        "nvidia_smi": run(
            "nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,"
            "memory.total,temperature.gpu,pstate --format=csv,noheader,nounits",
            20,
        ),
        "compute_apps": run(
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
            "--format=csv,noheader,nounits",
            20,
        ),
        "python_top": run(
            "ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu "
            "| grep -E 'python|vllm|ray' | grep -v grep | head -30",
            10,
        ),
        "load": run("uptime && free -h", 10),
    }


def main():
    ray.init(address="auto")
    ips = sorted({node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive")})
    refs = [probe.options(resources={f"node:{ip}": 0.001}).remote(ip) for ip in ips]
    for result in ray.get(refs, timeout=120):
        print(
            f"\n===== NODE {result['ip']} {result['hostname']} {result['date']} =====",
            flush=True,
        )
        for key in ("load", "nvidia_smi", "compute_apps", "python_top"):
            print(f"--- {key} ---", flush=True)
            print(result[key], flush=True)


if __name__ == "__main__":
    main()
