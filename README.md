# Is a Document Educational or Just Wikipedia-Style? — Pitfalls of Classifier-Based Quality Filtering

## Installation
The project dependencies are based on `vllm/vllm-openai:v0.11.2` Docker image.
Also, at times, scripts in the `scripts` directory include additional dependencies installed using `pip`.

## Usage 
See `scripts` directory for example usages
Directory `src` contains the source code used for the experiments.

## Citation
```bibtex
@inproceedings{klimaszewski-andruszkiewicz-2026-document,
    title = "Is a Document Educational or Just {W}ikipedia-Style? {---} Pitfalls of Classifier-Based Quality Filtering",
    author = "Klimaszewski, Mateusz  and
      Andruszkiewicz, Piotr",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-short.10/",
    pages = "99--108",
    ISBN = "979-8-89176-391-3",
    abstract = "Classifier-based Quality Filtering has recently emerged as a fundamental technique in constructing pre-training corpora. The ability to deploy a single model that can replace or supplement a set of heuristics has proven effective across numerous Large Language Models. In this work, we expose a critical vulnerability in this approach by demonstrating how a straightforward Wikipedia-style reformatting operation can substantially alter a model{'}s quality assessment and enable low-quality content to surpass filtering thresholds. Our analysis reveals that the FineWeb-Edu CQF model would reverse its filtering decision for approximately 7{\%} of evaluated documents, thereby admitting content into the pre-training corpus that would otherwise have been excluded."
}
```

## Acknowledgements

This research was funded in whole by the National Science Centre, Poland 2023/49/N/ST6/02691.
We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2025/018209.
