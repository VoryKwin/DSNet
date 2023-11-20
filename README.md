## DSNet:Double Strand Robotic Grasp Detection\\Network Based on Cross Attention

PyTorch implementation of paper "DSNet:Double Strand Robotic Grasp Detection\\Network Based on Cross Attention"
The article is currently under submission.

## Visualization of the architecture
![network_architecture.png](img%2Fnetwork_architecture.png)
<br>

## Datasets

Currently, both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php),
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/) and Multi Dataset are supported.

### Cornell Grasping Dataset
1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 

### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


2.  We use the setting in [here](https://github.com/ryanreadbooks/Modified-GGCNN) 


## Training

Training is done by the `main.py` script.  

Some basic examples:

```bash
# Train  on Cornell Dataset
python main.py   --dataset cornell

# k-fold training
python main_k_fold.py  --dataset cornell 


Trained models are saved in `output/models` by default, with the validation score appended.

## Visualize
Some basic examples:
```bash
# visualise grasp rectangles
python visualise_grasp_rectangle.py   --network your network address

# visualise heatmaps
python visualise_heatmaps.py  --network your network address

```

## Acknowledgement
Code heavily inspired and modified from https://github.com/dougsm/ggcnn.
