import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("WANDB_API_KEY")
url = "https://api.wandb.ai/graphql"

headers = {
    "Content-Type": "application/json"
}
auth = ("api", api_key)

query = """
query GetRuns($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 5) {
      edges {
        node {
          name
          state
          createdAt
          heartbeatAt
          updatedAt
          summaryMetrics
        }
      }
    }
  }
}
"""

payload = {
    "query": query,
    "variables": {
        "entity": "akshaykumarhudedmani-rl",
        "project": "goldenhour-rl"
    }
}

try:
    r = requests.post(url, auth=auth, headers=headers, json=payload, timeout=30)
    data = r.json()
    edges = data.get("data", {}).get("project", {}).get("runs", {}).get("edges", [])
    print(f"Total runs returned: {len(edges)}")
    for edge in edges:
        node = edge.get("node", {})
        name = node.get("name")
        state = node.get("state")
        heartbeat = node.get("heartbeatAt")
        created = node.get("createdAt")
        summary_str = node.get("summaryMetrics", "{}")
        summary = json.loads(summary_str) if isinstance(summary_str, str) else summary_str
        step = summary.get("train/global_step", summary.get("_step", "N/A"))
        reward = summary.get("train/reward", summary.get("reward", "N/A"))
        fmt = summary.get("train/rewards/format_reward_func/mean", "N/A")
        ver = summary.get("train/rewards/verifier_reward_func/mean", "N/A")
        loss = summary.get("train/loss", summary.get("loss", "N/A"))
        if state == "running":
            print("\n--- ACTIVE RUN DETAILS ---")
            print(json.dumps(summary, indent=2))
            print("--------------------------\n")
        print(f"Run: {name:<10} | State: {state:<10} | Step: {str(step):<4} | Created: {str(created):<20} | Heartbeat: {str(heartbeat):<20} | Reward: {reward} | Format: {fmt} | Verifier: {ver} | Loss: {loss}")
except Exception as e:
    print("Error:", e)
