### Rozum arm utils for running VIMA

Dependencies:
- Python 3.10
- `pip install -r requirements.txt`
- clone `utils` [repo](https://github.com/andrey1908/utils) @e301809 to `rozumarm_vima_utils/utils`
- clone `camera_utils` [repo](https://github.com/andrey1908/camera_utils) @96968ce to `rozumarm_vima_utils/camera_utils`
- clone `rozumarm_vima_cv` [repo](https://github.com/andrey1908/rozumarm_vima_cv) @cdc0733 to `rozumarm_vima_utils/rozumarm_vima_cv`

How to:
- to start (aruco_pos -> sim -> oracle -> arm) pipeline, run `scripts/run_aruco2sim_loop.py`
- to start (segmentation -> model -> arm) pipeline, run `scripts/run_model_loop.py`