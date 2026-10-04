# face_detection

A Siamese convolutional network for one-shot face verification, trained in a Colab notebook in TensorFlow. Given two face images, it predicts whether they show the same person.

Despite the repository name, this project does face verification, not face detection: it compares a captured face against stored reference images. The trained model is used by the desktop app in [mhussainahmad/kivy-app](https://github.com/mhussainahmad/kivy-app).

[Open the notebook in Colab](https://colab.research.google.com/github/mhussainahmad/face_detection/blob/main/siamese_model.ipynb)

## Overview

Everything is in `siamese_model.ipynb`:

1. **Data.**
   - Three folders: `anchor`, `positive` and `negative`.
   - Negatives are images from the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset.
   - Anchors and positives are 250x250 webcam crops of the person to be verified, collected with an OpenCV capture loop.
   - 300 images are taken from each folder.
2. **Pairs.** (anchor, positive) pairs are labelled 1 and (anchor, negative) pairs are labelled 0. Images are resized to 100x100 and scaled to [0, 1].
3. **Split.** 70% of pairs go to training and 30% to testing, in batches of 16.
4. **Model.**
   - The embedding network is adapted from Koch et al., "Siamese Neural Networks for One-shot Image Recognition."
   - It has four Conv2D layers (64, 128, 128 and 256 filters; kernels 10, 7, 4 and 4) with max-pooling between them, followed by a 4096-unit sigmoid dense layer.
   - A custom `L1Dist` layer takes the absolute difference of the two embeddings.
   - A single sigmoid unit outputs the similarity score.
   - Total parameters: 38,964,545.
5. **Training.** Custom `tf.GradientTape` training loop, binary cross-entropy, Adam (lr 1e-4), 50 epochs. Checkpoints are saved every 10 epochs.
6. **Evaluation and saving.** Precision and recall are computed on a test batch, and the model is saved as `siamesemodel.h5`.
7. **Verification.**
   - A webcam photo is captured in Colab and compared against every image in `application_data/verification_images/`.
   - A comparison counts as a match when its score is above a detection threshold.
   - The person is verified when the share of matches is above a verification threshold.

## Results

The notebook output reports precision 1.0 and recall 1.0. These are computed on a single batch of 16 test pairs, not the full test split. The test split itself is not fixed: the dataset is reshuffled on each iteration after caching, so these numbers are not a reliable held-out estimate.

## Getting started

Open the notebook in Colab using the link above.

- The notebook installs `tensorflow==2.4.1` and downloads LFW (`lfw.tgz`).
- The anchor/positive collection cell needs a local webcam. Run it locally, or upload your own `data/anchor` and `data/positive` images.
- The real-time verification cells use Colab's JavaScript webcam capture. They expect `application_data/input_image/` and `application_data/verification_images/` to exist.

## Repository layout

```
siamese_model.ipynb   Data preparation, model, training, evaluation, verification
LICENSE
```

The trained weights and image data are not committed.

## Tech stack

Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Google Colab.

## Credits

- Negative samples: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/), University of Massachusetts Amherst.
- Architecture: Koch, Zemel and Salakhutdinov, "Siamese Neural Networks for One-shot Image Recognition" (ICML Deep Learning Workshop, 2015).

## License

MIT. See [LICENSE](LICENSE).
