# K‑Means Clustering from Scratch  
K‑Means clustering is an unsupervised learning method that groups data points based on similarity.  
Using a synthetic dataset generated with `make_blobs(n_samples=500, centers=3, n_features=2, random_state=20)`, the algorithm predicts cluster membership by iteratively assigning points to the nearest centroid and updating centroid positions.  
The goal is to minimize intra‑cluster variance and reveal the natural grouping structure within the data.  
In conclusion, both Euclidean and Manhattan distance metrics produce evenly classified clusters, though centroid indices and plotted colors may differ due to random initialization.  

---

## Features  
- Generate synthetic dataset using `make_blobs` from scikit‑learn  
- Implement K‑Means clustering from scratch with helper functions  
- Support for multiple distance metrics (Euclidean & Manhattan)  
- Iterative centroid updates until convergence  
- Cluster visualization with matplotlib  

---

## Technologies  
- Python 3  
- scikit‑learn library  
- pandas library  
- numpy library  
- matplotlib library  

---

## How to Run  
1) **Open the notebook**  
   You can run the code either in Google Colab (recommended for easy setup) or in VS Code with the Jupyter Notebook extension installed.  

2) **Install dependencies**  
   Make sure you have the required libraries installed. If running locally, install them using:  
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```  

3) **Load the notebook**  
   Open the file `KMeansClusteringFromScratch.ipynb` in your chosen environment.  

4) **Run all cells**  
   - In Google Colab: click *Runtime > Run all*  
   - In VS Code: use the *Run All* option in the Jupyter Notebook toolbar  

5) **View results**  
   The notebook will display the clustering process step by step, showing centroid updates, scatter plots of clusters, and comparisons between Euclidean and Manhattan distance metrics.  
