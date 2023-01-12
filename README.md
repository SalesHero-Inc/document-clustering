# document-clustering
Repository for unsupervised document splitting to get document clusters

## Getting Started

To use this clustering tool in your project, first install the requirements using:

```bash
pip install -r requirements.txt
```


## Demo
Document Clustering:
```
from document_splitter.get_clusters import ClusterDetection
clusters = ClusterDetection().get_clustered_data(documents, "image")
```
