import time
import sys
import numpy as np
import faiss
import tritonclient.http as httpclient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
TRITON_URL = "triton-server:8000"
MODEL_NAME = "faiss_search"
INDEX_PATH = "/shared_data/faiss.index"


def wait_for_server():
    client = httpclient.InferenceServerClient(url=TRITON_URL)
    with console.status("[bold green]Waiting for Triton Server"):
        for _ in range(30):
            try:
                if client.is_server_live() and client.is_model_ready(MODEL_NAME):
                    console.print("[green]Server is ready[/green]")
                    return client
            except:
                pass
            time.sleep(1)
    console.print("[red]Server timeout[/red]")
    sys.exit(1)


def run_tests():
    client = wait_for_server()

    if not os.path.exists(INDEX_PATH):
        console.print(f"[red]Index file not found at {INDEX_PATH}[/red]")
        sys.exit(1)

    local_index = faiss.read_index(INDEX_PATH)

    console.rule(
        "[bold cyan]Verification (Identity Check)[/bold cyan]")

    target_id = 4242

    vector = local_index.reconstruct(target_id).reshape(1, 384)

    vector = vector.reshape(1, 384)

    top_k_data = np.array([[5]], dtype=np.int32)

    inputs = [
        httpclient.InferInput("QUERY_VECTOR", vector.shape, "FP32"),
        httpclient.InferInput("TOP_K", top_k_data.shape, "INT32")
    ]

    inputs[0].set_data_from_numpy(vector)
    inputs[1].set_data_from_numpy(top_k_data)

    start = time.time()
    res = client.infer(MODEL_NAME, inputs)
    dur = (time.time() - start) * 1000

    ids = res.as_numpy("NEIGHBOR_IDS")[0]

    table = Table(title=f"Query for Object #{target_id}")
    table.add_column("Rank")
    table.add_column("Returned ID")
    table.add_column("Match?", style="bold")

    match = False
    for i, idx in enumerate(ids):
        is_match = (idx == target_id)
        if is_match:
            match = True
        table.add_row(str(i+1), str(idx), "YES" if is_match else "NO")

    console.print(table)
    console.print(f"Latency: [yellow]{dur:.2f} ms[/yellow]")

    if ids[0] == target_id:
        console.print(
            Panel("[bold green]SUCCESS: Perfect Match! Score: 10/10[/bold green]"))
    else:
        console.print(
            Panel("[bold red]FAILURE: Wrong neighbor returned[/bold red]"))


if __name__ == "__main__":
    import os
    run_tests()
