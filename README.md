# 🚗⚡ GNN for Electric Vehicles  
**Graph Neural Networks for Smart Electric Vehicle Optimization**  

## 📌 Overview  
This repository hosts a collaborative project focused on developing **Graph Neural Networks (GNNs)** to optimize various aspects of electric vehicle (EV) systems. The goal is to leverage GNNs to model and enhance complex mobility-related challenges

## 📂 Repository Structure  
EV_GNN\
│── data --> Datasets for training and testing\
│──── \ ev --> EV datasets \
│──── \ ──── \ denmark --> EV dataset \
│──── \ ──── \ ──── \denmark_ev_station_availability\ --> Folder with EV files \
│──── \ ──── \ ──── \DenamarkEVstations.json --> EV metadata \
│──── \ other --> Observations maps \
│──── \ traffic --> Model checkpoints \
│──── \ ──── \ denmark --> Traffic dataset \
│──── \ ──── \ ──── \citypulse_traffic_raw_data_surrey_feb_jun_2014\ --> Folder with Traffic files \
│──── \ ──── \ ──── \ traffic_metadata.json --> Traffic metadata\
│── logs --> Folder for saving logs info \
│── env.yml --> Conda environment \
│── README.md --> Introduction and usage guide \
│── src\ --> Code \
│──── \ checkpoints --> Model checkpoints \
│──── \ config.py --> Code parameters \
│──── \ dataset --> Denmark dataset code \
│──── \ main.py --> Start training \
│──── \ model --> Collection of Neural Models  \
│──── \ utils --> Utils functions \
│──── \ webscraping --> Gathering denmark dataset data \

## 🚀 Run the code
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-org/EV_GNN.git
   cd EV_GNN
   ```
2. Install the dependencies:
    ```bash
    conda env create -f env.yml
    conda activate tf_env
    ```
   
3. Run training:
   ```bash
    python main_example.py
    ```
