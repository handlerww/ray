"""Small cluster training

This training run will start 4 workers on 4 nodes (including head node).

Test owner: krfricke

Acceptance criteria: Should run through and report final results.
"""
import json
import os
import time

import ray

import glob
import os
import time

import ray

from xgboost_ray import train, RayDMatrix, RayFileType, \
    RayDeviceQuantileDMatrix

def train_ray(path,
              num_workers,
              num_boost_rounds,
              num_files=0,
              regression=False,
              use_gpu=False,
              ray_params=None,
              xgboost_params=None,
              **kwargs):
    # path = os.path.expanduser(path)
    # if not os.path.exists(path):
    #     raise ValueError(f"Path does not exist: {path}")

    if num_files:
        files = sorted(glob.glob(f"{path}/**/*.parquet"))
        while num_files > len(files):
            files = files + files
        path = files[0:num_files]

    use_device_matrix = False
    if use_gpu:
        try:
            import cupy  # noqa: F401
            use_device_matrix = True
        except ImportError:
            use_device_matrix = False

    if use_device_matrix:
        dtrain = RayDeviceQuantileDMatrix(
            path,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)
    else:
        dtrain = RayDMatrix(
            path,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)

    config = {"tree_method": "hist" if not use_gpu else "gpu_hist"}

    if not regression:
        # Classification
        config.update({
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        })
    else:
        # Regression
        config.update({
            "objective": "reg:squarederror",
            "eval_metric": ["logloss", "rmse"],
        })

    if xgboost_params:
        config.update(xgboost_params)

    start = time.time()
    evals_result = {}
    additional_results = {}
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        additional_results=additional_results,
        num_boost_round=num_boost_rounds,
        ray_params=ray_params or RayParams(
            max_actor_restarts=2,
            num_actors=num_workers,
            cpus_per_actor=1,
            gpus_per_actor=1 if not use_gpu else 1),
        evals=[(dtrain, "train")],
        **kwargs)
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    out_file = os.path.expanduser(
        "~/benchmark_{}.xgb".format("cpu" if not use_gpu else "gpu"))
    bst.save_model(out_file)

    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))
    return bst, additional_results, taken

if __name__ == "__main__":
    os.environ["RXGB_PLACEMENT_GROUP_TIMEOUT_S"] = "1200"

    addr = os.environ.get("RAY_ADDRESS")
    job_name = os.environ.get("RAY_JOB_NAME", "train_gpu_connect")
    if addr.startswith("anyscale://"):
        ray.init(address=addr, job_name=job_name)
    else:
        ray.init(address="auto")

    from xgboost_ray import RayParams
    #from ray.util.xgboost.release_test_util import train_ray

    ray_params = RayParams(
        elastic_training=False,
        max_actor_restarts=2,
        num_actors=4,
        cpus_per_actor=4,
        gpus_per_actor=1)



    start = time.time()

    train_ray(
        path="/data/classification.parquet",
        num_workers=4,
        num_boost_rounds=100,
        num_files=25,
        regression=False,
        use_gpu=True,
        ray_params=ray_params,
        xgboost_params=None,
    )

    taken = time.time() - start

    result = {
        "time_taken": taken,
    }
    test_output_json = os.environ.get("TEST_OUTPUT_JSON",
                                      "/tmp/train_gpu_connect.json")
    with open(test_output_json, "wt") as f:
        json.dump(result, f)

    print("PASSED.")
