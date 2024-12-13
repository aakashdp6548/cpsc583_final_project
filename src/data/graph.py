from torch_geometric.nn import radius_graph, knn_graph

class SpatialGraphConstructor:
    def __init__(self, k_neighbors=10):
        """Initialize graph constructor with knn initialization.
        
        Args:
            k_neighbors (int): Number of neighbors for k-NN graph
            radius (float, optional): Radius for radius-based graph
        """
        self.k = k_neighbors
    
    def build_graph(self, spatial_coords):
        """Build graph from spatial coordinates.
        
        Args:
            spatial_coords (torch.Tensor): [N x 2] tensor of spatial coordinates
            
        Returns:
            torch.Tensor: [2 x E] edge index tensor
        """
        return knn_graph(spatial_coords, k=self.k)
