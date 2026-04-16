import os 
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def export_dataset_to_csr(dataset_name="Cora", out_dir="data"):
    print(f"Fetching {dataset_name} dataset...")
    dataset = Planetoid(root=f'./tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]
    print("Converting to CSR format...")

    adj_sparse = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()

    dataset_dir = os.path.join(out_dir, dataset_name.lower())
    os.makedirs(dataset_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "meta.txt"), "w") as f:
        f.write(f"{data.num_nodes} {adj_sparse.nnz}\n")
    
    np.savetxt(os.path.join(dataset_dir, "row_ptr.txt"), adj_sparse.indptr, fmt="%d")
    np.savetxt(os.path.join(dataset_dir, "col_idx.txt"), adj_sparse.indices, fmt="%d")

    print(f"Succesfully exported {dataset_name} dataset to {dataset_dir}\n")    
    print(f"Dataset stats: num_nodes={data.num_nodes}, num_edges={adj_sparse.nnz}, avg_degree={adj_sparse.nnz/data.num_nodes:.2f}")

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    for dataset_name in ["Cora", "Citeseer"]: 
        export_dataset_to_csr(dataset_name)

   

