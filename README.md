# SignBridge AI

SignBridge AI is a research prototype for Greek Sign Language. It recognises isolated signs from webcam or uploaded video in the browser and provides a library of recorded signs and phrases.

- [Live application](https://signbridge-ai-gsl.pages.dev/#/try)
- [Application source](web/src) and [training and evaluation code](ml)
- [Original technical guide](PROJECT_GUIDE.md)
- [Colab instructions](COLAB.md)

To inspect the source in [Google Colab](https://colab.research.google.com/), download [the notebook](colab/SignBridge-Colab.ipynb) and [the matching ZIP](colab/SignBridge-Colab-Source.zip) from GitHub. Open the notebook with **File → Upload notebook**, run its first cell, and choose the ZIP. The notebook uses no API keys. The second cell runs a small set of checks for the evaluation code using synthetic inputs.

This repository contains source and instructions. A full local app build and model training require separate datasets, model files, and dependencies. Upstream license notices are in [web/deployment/licenses](web/deployment/licenses).
