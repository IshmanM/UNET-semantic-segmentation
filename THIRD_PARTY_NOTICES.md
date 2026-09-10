# Third-Party Software and Data

U-Net Semantic Segmentation uses third-party open-source software through
external Python dependencies. These packages are not authored by this project
and remain subject to their respective upstream licenses.

Major direct dependencies used by the source code include:

- PyTorch: BSD-3-Clause
- Torchvision: BSD-3-Clause
- NumPy: BSD-3-Clause
- pandas: BSD-3-Clause
- Pillow: HPND
- patchify: MIT
- python-dotenv: BSD-3-Clause
- tqdm: upstream MIT / MPL-2.0 licensing
- TensorBoard: Apache-2.0

The repository's historical `requirements.txt` also contains transitive and
environment-specific packages. Those packages are installed as external
dependencies and remain governed by their own upstream licenses.

## Semantic Drone Dataset

This project was trained and evaluated using the Semantic Drone Dataset from
the Institute of Computer Graphics and Vision at Graz University of Technology:

https://ivc.tugraz.at/research-project/semantic-drone-dataset/

The dataset is provided for non-commercial use and requires attribution.
Its terms prohibit redistribution of the dataset or modified versions of the
dataset. They permit distribution of abstract derivative representations, such
as trained models and additional annotations that do not directly include or
allow recovery of the dataset, but those derivative works also remain subject
to the dataset's non-commercial restriction.

The Semantic Drone Dataset itself is not licensed under this project's MIT
License.

Trained model checkpoints and dataset-derived outputs or evaluation artifacts
included in this repository are excluded from the project's MIT License to the
extent that the Semantic Drone Dataset terms apply to them. They should be used
only in accordance with the dataset's terms.

Raw Semantic Drone Dataset images should not be redistributed in this
repository.

## U-Net Architecture Reference

The model architecture was implemented with reference to:

Ronneberger, O., Fischer, P., and Brox, T. (2015).
"U-Net: Convolutional Networks for Biomedical Image Segmentation."

https://arxiv.org/abs/1505.04597

The architecture described by the paper is used as a technical reference. The
source code in this repository is independently implemented. Figures from the
paper are not covered by this project's MIT License and should not be
redistributed unless separate permission or a suitable license applies.

## Other Learning References

The repository also cites Ekin Tiu's article, "Metrics to Evaluate your
Semantic Segmentation Model," as a learning reference for segmentation metrics:

https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

If third-party source code, datasets, images, models, or other assets are later
copied or vendored directly into this repository, their required copyright
notices, attribution, and license terms should be preserved.
