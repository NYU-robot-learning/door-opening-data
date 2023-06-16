# Door Opening Dataset

This is a repository containing door-opening data using a modified reacher-grabber stick with iPhone mounted on it. This data was used in works **FISH**: [[Arxiv]](https://arxiv.org/abs/2303.01497) [[Project page and videos]](https://fast-imitation.github.io/) and **VINN**: [[Arxiv]](https://arxiv.org/abs/2112.01511) [[Project page and videos]](https://jyopari.github.io/VINN/). The data can be downloaded from: https://drive.google.com/file/d/1CRfvW9sLZMH_UJwf8ILEbu9fkgABarrJ/view?usp=sharing


The following will load the data in numpy and save it as npy file locally. NOTE: The npy file can become extremely big because the images are loaded in full resolution. So edit the code to either fragment the data and save it in chunks or downsample images before saving them.
```
python loader.py
```


## Bibtex
```
@article{haldar2023teach,
         title={Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations},
         author={Haldar, Siddhant and Pari, Jyothish and Rai, Anant and Pinto, Lerrel},
         journal={arXiv preprint arXiv:2303.01497},
         year={2023}
}

@misc{VINN,
  author = {Pari, Jyo and Shafiullah, Mahi and Arunachalam, Sridhar and Pinto, Lerrel},
  title = {Visual Imitation through Nearest Neighbors (VINN) implementation},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jyopari/VINN/tree/main}},
}
```

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
