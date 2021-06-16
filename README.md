# log-placement

Deep learning framework implemented as described in this study: `Li, Z., Chen, T. H., and Shang, W. (2020, September). Where shall we log? studying and suggesting logging locations in code blocks. In 2020 35th IEEE/ACM International Conference on Automated Software Engineering (ASE) (pp. 361-372). IEEE.`

## Dataset

See example datasets for correct format.

Place your dataset as `data/processed/dataset.json`.

Example datasets were generated from the CloudStack repository.

## Run experiments

- `sudo docker build --tag log-placement .`

- `sudo docker run log-placement`