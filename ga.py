from pathlib import Path
import fastapi
import fastapi.staticfiles
import modal
import random
from population import Population
from numba import jit
from fastapi import Query

stub = modal.Stub("mattemix-solver")

image = (
    modal.Image.conda()
    .conda_install(["numba"], channels=["conda-forge"])
    .pip_install("numpy")
)

stub.dict = modal.Dict()

web_app = fastapi.FastAPI()


@stub.function(image=image, mounts=modal.create_package_mounts(["population"]))
def find_solution(seed, call_id: str):
    from modal import container_app
    import time
    import numpy as np
    import time
    from population import Population

    np.random.seed(seed)

    # Get the parameters from the shared dictionary
    init_dict = container_app.dict[call_id].copy()
    print("Initial update_dict:")
    print(init_dict)
    size = init_dict.get("pop_size")
    max_time = init_dict.get("max_time")
    dice_roll = init_dict.get("dice_array")
    mut_rate = init_dict.get("mut_rate")
    cross_rate = init_dict.get("cross_rate")
    elite_rate = init_dict.get("elite_rate")
    # Convert comma separated string to list of ints
    dice_roll = [int(x) for x in dice_roll.split(",")]
    print(dice_roll)
    pop = Population(
        size,
        dice_throw=dice_roll,
        mut_rate=mut_rate,
        cross_rate=cross_rate,
        elite_rate=elite_rate,
    )

    start_time = time.time()
    while time.time() - start_time < max_time:
        pop.evolve()
        score = pop.best_fitness
        solution = pop.best_solution.tolist()
        correct_equations = pop.correct_equations.tolist()
        update_dict = container_app.dict[call_id].copy()
        if score >= update_dict.get("best_score"):
            update_dict.update(
                {
                    "best_score": score,
                    "best_solution": solution,
                    "correct_equations": correct_equations,
                }
            )
        update_dict.update(
            {
                "time_left": int(max([max_time - (time.time() - start_time), 0])),
                "num_tested": update_dict.get("num_tested") + pop.generation * size,
            }
        )
        container_app.dict.put(key=call_id, value=update_dict)
    print("Finished")
    update_dict["status"] = "finished"
    update_dict["time_left"] = 0
    container_app.dict.put(key=call_id, value=update_dict)
    test_dict = container_app.dict[call_id].copy()
    print("Final test_dict:")
    print(test_dict)
    return


@web_app.get("/solve")
async def init_solve_job(
    dice_array: str = Query(
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14",
        description="Comma-separated list of 14 integers between 1 and 15",
        regex=r"^(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9]),(1[0-5]|[1-9])$",
    ),
    pop_size: int = Query(50000, ge=1000, le=100000),
    timeout: int = Query(60, ge=1, le=60),
    num_populations: int = Query(1, ge=1, le=3),
    mut_rate: float = Query(0.05, ge=0.01, le=0.9),
    cross_rate: float = Query(0.1, ge=0.05, le=0.9),
    elite_rate: float = Query(0.1, ge=0.05, le=0.9),
):
    # Spawn a call to the function
    from modal import container_app
    from uuid import uuid4
    from population import check_solution

    # Generate a unique call id
    call_id = str(uuid4())
    # Initialize a shared dict with the call_id
    print(call_id)
    solution = [int(x) for x in dice_array.split(",")]
    initial_score, correct_equations = check_solution(solution)
    call_dict = {
        "dice_array": dice_array,
        "best_score": initial_score,
        "best_solution": solution,
        "correct_equations": correct_equations,
        "status": "running",
        "pop_size": pop_size,
        "num_tested": 0,
        "max_time": timeout,
        "time_left": timeout,
        "num_populations": num_populations,
        "mut_rate": mut_rate,
        "cross_rate": cross_rate,
        "elite_rate": elite_rate,
    }
    container_app.dict[call_id] = call_dict
    # Make sure we include 42 as a seed
    seeds = [42 + i for i in range(num_populations)]
    calls = [find_solution.spawn(seed=s, call_id=call_id) for s in seeds]

    return {"call_id": call_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    # Poll the results written to modal Dict for a given call_id
    from modal import container_app
    import time

    caller_dict = container_app.dict.get(call_id)
    return caller_dict


assets_path = Path(__file__).parent / "stlite"


@stub.asgi(image=image, mounts=[modal.Mount("/assets", local_dir=assets_path)])
def wrapper():
    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))
    return web_app


if __name__ == "__main__":
    random.seed(42)
    print("Starting...")
    stub.serve()
