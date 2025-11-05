<!-- Template source: See: https://github.com/othneildrew/Best-README-Template -->
<a id="readme-top"></a>

[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/arthur-testard/)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/art-test-stack/my-gpt">
    <img src="rsc/logo.jpg" alt="Logo" height="350">
  </a> -->

<h3 align="center">Small Language Model</h3>

  <p align="center">
    This project is the implementation of a little GPT model trained on different datasets
    <br />
    <a href="https://github.com/art-test-stack/my-gpt"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/art-test-stack/my-gpt/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#the-implementation">The implementation</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#sources">Sources</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details> -->



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- ### The implementation -->


### Built With

* [![Torch][Torch]][Torch-url]



<!-- GETTING STARTED -->
<!-- ## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:art-test-stack/my-gpt.git
   ```
2. Create a virtual environment
    
    For example I use [virtualenv](https://virtualenv.pypa.io/en/latest/):
   ```sh
   virtualenv -p python 3.10 venv
   ```
3. Install pip packages
   ```sh
   pip install -r requirements.txt
   ``` -->



<!-- ## Usage -->




<!-- ROADMAP -->
<!-- ## Roadmap -->

<!-- - [ ]  -->


<!-- See the [open issues](https://github.com/art-test-stack/my-gpt/issues) for a full list of proposed features (and known issues). -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request -->

## Roadmap 

* Tokenization 
  - BPE implementation in Python
  - Rust implementation
* Positional embedding
  - Absolute
  - rotary
* Transformer
  - Attention mechanism
  - Multihead attention
  - flash attention
  - FFN, RMSNorm layers
* Training
  - Pre-training
  - fine-tuning
  - intruction tuning
  - rlhf, dpo
  - ddp, fsdp method
* Sampling
  - temperature
  - top-k, top-p
  - beam-search
* Too move beyond
  - KV-cache
  - sliding window
  - memory layers?
  - MoE
  - Quantization
* Training on Synthetic Data
  - generate data
  - model teacher


## Data

<table>
    <thead>
        <tr>
            <th align="center">Source</th>
            <th align="center">Documents</th>
            <th align="center">Tokens</th>
            <th align="center">Ratio</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="left"><b><a href="https://wikipedia.org/">Wikipedia</a></b></td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td align="left"><b><a href="https://huggingface.co/datasets/Skylion007/openwebtext">OpenWebText</a></b></td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <th align="left">Total</th>
            <th align="center">-</th>
            <th align="center">-</th>
            <th align="center">100.00 %</th>
        </tr>
    </tbody>
</table>

<!-- Sources -->
## Sources

1. [Attention is all you need](https://arxiv.org/pdf/1706.03762)
2. [Building a text generation model from scratch by Vincent Bons](https://wingedsheep.com/building-a-language-model/)
3. [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT/tree/master) 
4. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
5. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Arthur Testard - testardarthur@gmail.com

Project Link: [https://github.com/art-test-stack/my-gpt](https://github.com/art-test-stack/my-gpt)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/art-test-stack/my-gpt.svg?style=for-the-badge
[contributors-url]: https://github.com/art-test-stack/my-gpt/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/art-test-stack/my-gpt.svg?style=for-the-badge
[forks-url]: https://github.com/art-test-stack/my-gpt/network/members
[stars-shield]: https://img.shields.io/github/stars/art-test-stack/my-gpt.svg?style=for-the-badge
[stars-url]: https://github.com/art-test-stack/my-gpt/stargazers
[issues-shield]: https://img.shields.io/github/issues/art-test-stack/my-gpt.svg?style=for-the-badge
[issues-url]: https://github.com/art-test-stack/my-gpt/issues
[license-shield]: https://img.shields.io/github/license/art-test-stack/my-gpt.svg?style=for-the-badge
[license-url]: https://github.com/art-test-stack/my-gpt/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/arthur-testard
[product-screenshot]: images/screenshot.png
[Torch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Torch-url]: https://pytorch.org/
