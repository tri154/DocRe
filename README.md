# Requirements
To install requirements:
```bash
pip install -r requirements.txt
```
# Training:
Move to `src` directory:
```bash
cd src
```
To train the model on dataset CDR, GDA and BioRed and, run this command:
```bash
bash evaluate_cdr.sh
bash evaluate_gda.sh
bash evaluate_biored.sh
```
Each script trains the model in 5 seeds, the output is saved in `src/logs` directory, make sure to clear the `src/logs` and `src/outputs` before training on different datasets.
# Example:
Training with CDR 
[link](https://www.kaggle.com/code/saverysad/exper9-sota?scriptVersionId=255678134)

# Result:
| Model                         | CDR | GDA | BioRED| 
|------------------------------|------------------|------------------|------------------|
| HTGRS            | 86.9         | 82.2         | 66.9        | 
| PSD_ATLOP             | 69.52         | 84.45        |  N/A |
| Topic-BiGRU-U-Net | 87.1        | 84.1          | N/A          | 
| ours              | **87.25 +- 0.4**       | **84.52 +- 0.2**       | **73.27 +- 1.2**       | 
