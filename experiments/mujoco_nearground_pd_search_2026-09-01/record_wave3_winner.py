"""Record the exact 1 kHz nominal trajectory for the selected wave=3 point."""
import json

from run_search import OUT, run_one


candidate = json.loads(
    (OUT / "common/basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json").read_text()
)["config"]


if __name__ == "__main__":
    row = run_one((candidate, 3.0, 6.0, 1.0, True, "common"))
    print(
        json.dumps(
            {
                "id": row["id"],
                "D": row["metrics"]["displacement_mm"],
                "trace_recorded": True,
            }
        )
    )
