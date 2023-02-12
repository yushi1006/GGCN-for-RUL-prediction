# GGCN-for-RUL-prediction
Graph Convolutional Network for RUL prediction with multi-sensor signals
This is the python code for the paper:"A gated graph convolutional network with multi-sensor signals for remaining useful life prediction
https://doi.org/10.1016/j.knosys.2022.109340"
If this code helps you, please cite this paper.
#################################################################################################################
A gated graph convolutional network (GGCN) is developed for multi-sensor signal fusion and RUL prediction. Firstly, spatial–temporal graphs
are constructed from multi-sensor signals as input of the prognosis model. Next, gated graph convolutional layers are built to accurately 
extract degradation features by simultaneously modeling the temporal and spatial dependencies in multi-sensor signals. Finally, the extracted
features are fed into a quantile regression layer to estimate the RUL and its confidence interval.
