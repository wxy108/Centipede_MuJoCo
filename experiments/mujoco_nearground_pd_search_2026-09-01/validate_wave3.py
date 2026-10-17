"""Exact 12 s and half-timestep validation for the best tested wave=3 point."""
import concurrent.futures
import json

from run_search import OUT, run_one


candidate = json.loads(
    (OUT / "common/basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json").read_text()
)["config"]


if __name__ == "__main__":
    tasks = [
        (candidate, 3.0, 12.0, 1.0, True, "common"),
        (candidate, 3.0, 6.0, 0.5, True, "common"),
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run_one, tasks))
    (OUT / "wave3_validation_ids.json").write_text(
        json.dumps([row["id"] for row in rows], indent=2)
    )
    print(
        json.dumps(
            [
                {
                    "id": row["id"],
                    "D": row["metrics"]["displacement_mm"],
                    "failed": [key for key, value in row["gates"].items() if not value],
                }
                for row in rows
            ]
        )
    )
