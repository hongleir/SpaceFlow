import scanpy as sc
import squidpy as sq
from SpaceFlow import SpaceFlow

adata = sq.datasets.seqfish()
sf = SpaceFlow.SpaceFlow(expr_data=adata.X, spatial_locs=adata.obsm['spatial'])
sf.preprocessing_data()
sf.train()
sf.segmentation()
sf.plot_segmentation()
sf.pseudo_Spatiotemporal_Map()
sf.plot_pSM()