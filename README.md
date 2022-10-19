# Decoupled-Unbiased-Teacher

#### Installation & Data Prepare

Please check [INSTALL.md](INSTALL.md) for installation instructions. Please prepare the data following [this repo](https://github.com/CityU-AIM-Group/SFPolypDA)

#### Training

The following command line controls different stages of training:

```bash
# Train the source only model
python tools/train_net_mcd.py --config-file configs/sf/source_only.yaml SOLVER.SFDA_STAGE 1

# Train the model with DUT on Abnormal Symptoms dataset
python tools/train_net_mcd.py --config ./configs/sf/dut_hcmus.yaml OUTPUT_DIR outputs/dut_hcmus

# Train the model with DUT on WCE dataset
python tools/train_net_mcd.py --config ./configs/sf/dut_hcmus.yaml OUTPUT_DIR outputs/dut_hcmus
```

#### Ackonwledgement

The code is based on FCOS. For enquiries please contact xliu423-c@my.cityu.edu.hk.