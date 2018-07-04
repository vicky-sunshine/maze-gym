# maze-gym

Learn how to build a open AI gym enviroment.

Since I focus on building env first, so the learnining algorithm and run script is currently from [learning maze of MorvanZhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/2_Q_Learning_maze).

## Environment

```shell
# python 3.6
gym==0.10.5
numpy==1.14.3
pandas==0.23.0
```

## Maze map

Below is a 3 x 3 map:

```csv
0,0,0
0,2,0
2,3,0
```

- `0`: walkable road (reward 0)
- `2`: falling hole (reward -1, and stop walking)
- `3`: treasure (reward 1, and stop walking)

The adventurer will start at (1, 0), which means the second row and the first column
The program will show `1` for adventurer.

Example:
```csv
0,0,0
1,2,0
2,3,0
```

You can specify the map config at `MAZE_MAP_PATH` variable in [maze_env.py](./maze_env.py).

## Run

Just run

```shell
python run_this.py
```

and it will walk!

Example console output:
```
======
choose action: up
[[0 1 0 2]
 [0 2 0 2]
 [2 0 0 0]
 [3 0 2 2]]
user loc: (0, 1)
======
choose action: right
[[0 0 1 2]
 [0 2 0 2]
 [2 0 0 0]
 [3 0 2 2]]
user loc: (0, 2)
======
choose action: left
[[0 1 0 2]
 [0 2 0 2]
 [2 0 0 0]
 [3 0 2 2]]
user loc: (0, 1)

...
```

