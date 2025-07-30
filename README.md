[![arXiv](https://img.shields.io/badge/arXiv-2507.16379-b31b1b.svg)](https://arxiv.org/abs/2507.16379)
# ApproxGNN: A Pretrained GNN for Parameter Prediction in Design Space Exploration for Approximate Computing

This repository contains data and code associated with our paper presented at the 2025 IEEE/ACM International Conference on Computer Aided Design (ICCAD). A [preprint is available on arXiv](https://arxiv.org/abs/2507.16379). If you find this work useful, please cite it as follows:

_VLCEK, O.; MRAZEK, V. ApproxGNN: A Pretrained GNN for Parameter Prediction in Design Space Exploration for Approximate Computing. 2025 IEEE/ACM International Conference on Computer Aided Design (ICCAD), Munich, 2025._

```bibtex
@inproceedings{vlcek:iccad25,
    author    = "Ondrej Vlcek and Vojtech Mrazek",
    title     = "ApproxGNN: A Pretrained GNN for Parameter Prediction in Design Space Exploration for Approximate Computing",
    booktitle = "2025 IEEE/ACM International Conference on Computer Aided Design (ICCAD)",
    year      = "2025",
    pages     = "8",
    address   = "Munich"
}
```


# Repository description
In this work we create ML model based on Graph Neural Networks to estimate QoR and HW cost of approximate accelerators (in this case for the image kernel filters). The architectures of the proposed GNNs are available in [models.py](approxgnn/models.py). This repository provides the core algorithms and neural network models used to build the ApproxGNN predictor. All source code can be found in the [approxgnn](approxgnn) directory.

Pretrained models and training datasets are available. Example files are located in the [data](data) and [pretrained](pretrained) directories. The fully characterized approximate components are available in [components](components/) folder. These components are taken from [EvoApproxLib library](https://github.com/ehw-fit/evoapproxlib).

# Usage
For usage, Python version >= 3.10 is required. For instalation use the requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```