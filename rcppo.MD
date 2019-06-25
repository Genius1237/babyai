## Instructions to run code with Reverse Curriculum Learning
1. Generate demos using the bot and let it store them in the default location
```
scripts/make_agent_demos.py --env ENV_NAME
```
By default, 1000  demos are generated. All the demos that are generated are used

2. Run the training script using `scripts/train_rcppo.py`
```
scripts/train_rcppo.py --env ENV_NAME
```
There are a few more flags on top of the flags in `train_rl.py`: 
* `--update-frequency`: Number of frames after which a condition is checked to determine whether to update curriculum. Default is 10 (which is same as log interval)
* `--es-method`: Either 1 or 2. Determines the variant of the test condition used. Default is 1
* `--curr-method`: Any integer, or the string log. Determines the factor by which the curriculum length is reduced. Default is 1 (no reduction)
* `--curr-memory`: Enables memory for observations from demos (only for branch rcppo-memory)

## Warnings
1. By default, `--procs` is set to 64, which leads to high memory usage. Run the code with atleast 32 GB of RAM, or reduce the number of procs
2. When `--curr-memory` is set, a significant amount of GPU memory is also needed, and in some cases the code where the demos are very long, you'll run out of video memory even on a GPU with 16GB of VRAM