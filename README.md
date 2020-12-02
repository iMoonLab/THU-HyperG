[license-image]: https://img.shields.io/github/license/iMoonLab/THU-HyperG.svg
[license-url]: https://github.com/iMoonLab/THU-HyperG/blob/master/LICENSE

THU-HyperG: A python toolbox for hypergraph learning  
===


[![LICENSE Version][license-image]][license-url]




Introduction
---
**THU-HyperG** is a python toolbox for hypergraph learning. Hypergraph is a generalization of graph, which is composed of a set of nodes and a set of hyperedges. Different from simple graph, in which each edge connect a pair of nodes, each hyperedge can connect any number of nodes in hypergraph. The flexible edge degree in hypergraph enables the hypergraph model to formulate the high-order correlation of data.

Hypergraph learning is mainly consisted of two procedures, hypergraph generation and learning on hypergraph. Thus, in this toolbox, we provide several hypergraph generation methods and learning methods on hypergraph. Some useful utilities are also included in this toolbox. Besides, we provide some examples about how to apply this toolbox to the tasks like classification and segmentation.  

Specifically, it provides the following functionalities:
* **Hypergraph generation methods**: knn based, epsilon ball based, clustering based, l1 representation based and grid based methods.
* **Learning on hypergraph methods**: transductive learning, inductive learning, diffusion and clustering methods.
* **Utils**: some preprocessing and evaluation methods.

Installation
---
You can install from source
    
    git clone https://github.com/iMoonLab/THU-HyperG.git
    cd THU-HyperG
    pip install .

Examples
---

We provide some [examples](https://github.com/iMoonLab/THU-HyperG/tree/master/examples) in the examples/ directory about how to apply this toolbox to the tasks like classification and segmentation.  

Citing
--- 
if you find our work useful in your research, please consider citing:

@article{gao2020hypergraph,
  title={Hypergraph Learning: Methods and Practices.},
  author={Gao, Y and Zhang, Z and Lin, H and Zhao, X and Du, S and Zou, C},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}

Contributing
---
We appreciate all contributions to improve THU-HyperG. Pull requests are always welcomed.
