import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# configuration
BASE_PATH = r"C:\Users\sango\OneDrive\Desktop\Istudio"
DATA_DIR = os.path.join(BASE_PATH, "dataset", "dataset", "faces")
SAVE_DIR = os.path.join(BASE_PATH, "results")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

img_shape = (100, 100)
test_split = 0.40
seed = 42
np.random.seed(seed)

# imposters for testing
holdout_classes = ['Disha', 'Farhan']

def load_data(data_path, shape, holdouts):
    faces = []
    labels = []
    imp_faces = []
    imp_labels = []
    
    classes = sorted(os.listdir(data_path))
    
    for c in classes:
        dir_path = os.path.join(data_path, c)
        if not os.path.isdir(dir_path):
            continue
            
        imgs = sorted(glob.glob(os.path.join(dir_path, '*.*')))
        valid_exts = ('.jpg', '.jpeg', '.png', '.pgm', '.bmp')
        imgs = [p for p in imgs if p.lower().endswith(valid_exts)]
        
        for img_path in imgs:
            try:
                # read and resize
                im = Image.open(img_path).convert('L')
                im = im.resize(shape, Image.LANCZOS)
                vec = np.array(im, dtype=np.float64).flatten()
                
                if c in holdouts:
                    imp_faces.append(vec)
                    imp_labels.append(c)
                else:
                    faces.append(vec)
                    labels.append(c)
            except Exception as e:
                pass

    if len(faces) == 0:
        raise ValueError("No enrolled faces found! Check path.")
        
    X = np.column_stack(faces)
    X_imp = np.column_stack(imp_faces) if len(imp_faces) > 0 else np.array([])
    enrolled = [c for c in classes if c not in holdouts]
    
    return X, labels, X_imp, imp_labels, enrolled

print("Loading dataset...")
X_all, y_all, X_imp, y_imp, valid_classes = load_data(DATA_DIR, img_shape, holdout_classes)

print(f"Loaded {X_all.shape[1]} enrolled images and {X_imp.shape[1]} imposter images.")

le = LabelEncoder()
y_all_num = le.fit_transform(y_all)

# save a plot showing one image per class
print("Saving sample images...")
fig, ax = plt.subplots(1, len(valid_classes), figsize=(12, 3))
for i, cls in enumerate(valid_classes):
    idx = y_all.index(cls)
    img = X_all[:, idx].reshape(img_shape[1], img_shape[0])
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(cls)
    ax[i].axis('off')
plt.savefig(os.path.join(SAVE_DIR, "samples.png"))
plt.close()

# train test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
train_idx, test_idx = next(sss.split(X_all.T, y_all_num))

X_train = X_all[:, train_idx]
y_train_num = y_all_num[train_idx]
y_train = [y_all[i] for i in train_idx]

X_test = X_all[:, test_idx]
y_test_num = y_all_num[test_idx]
y_test = [y_all[i] for i in test_idx]

print(f"Train size: {X_train.shape[1]}, Test size: {X_test.shape[1]}")

print("Running PCA...")
# 1. compute mean face
mean_face = X_train.mean(axis=1, keepdims=True)

# 2. subtract mean
phi = X_train - mean_face

# 3. covariance matrix C = A^T * A
cov = np.dot(phi.T, phi)

# 4. eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(cov)

# sort descending
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# keep positive eigenvalues
pos_idx = eig_vals > 1e-6
eig_vals = eig_vals[pos_idx]
eig_vecs = eig_vecs[:, pos_idx]

# 5. compute eigenfaces U = A * V
eigenfaces = np.dot(phi, eig_vecs)

# normalize eigenfaces
for i in range(eigenfaces.shape[1]):
    eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

# project train and test
omega_train = np.dot(eigenfaces.T, phi)
omega_test = np.dot(eigenfaces.T, X_test - mean_face)
omega_imp = np.dot(eigenfaces.T, X_imp - mean_face)

# save mean face
plt.imshow(mean_face.reshape(img_shape[1], img_shape[0]), cmap='gray')
plt.title('Mean Face')
plt.axis('off')
plt.savefig(os.path.join(SAVE_DIR, 'mean_face.png'))
plt.close()

# save top eigenfaces
fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    if i < eigenfaces.shape[1]:
        ef = eigenfaces[:, i].reshape(img_shape[1], img_shape[0])
        ax[i//5, i%5].imshow(ef, cmap='gray')
        ax[i//5, i%5].set_title(f"PC {i+1}")
    ax[i//5, i%5].axis('off')
plt.savefig(os.path.join(SAVE_DIR, 'eigenfaces.png'))
plt.close()

# plot explained variance
cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)
plt.plot(cum_var)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(0.95, color='r', linestyle='--')
plt.savefig(os.path.join(SAVE_DIR, 'variance.png'))
plt.close()

print("Evaluating ANN across different k values...")
k_vals = list(range(1, len(eig_vals)+1, max(1, len(eig_vals)//20)))
acc_scores = []

for k in k_vals:
    # use first k components
    X_tr_k = omega_train[:k, :].T
    X_te_k = omega_test[:k, :].T
    
    # train simple ANN
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=seed, early_stopping=True)
    clf.fit(X_tr_k, y_train_num)
    
    pred = clf.predict(X_te_k)
    acc = accuracy_score(y_test_num, pred)
    acc_scores.append(acc)

best_k = k_vals[np.argmax(acc_scores)]
best_acc = max(acc_scores)
print(f"Best k: {best_k} with accuracy: {best_acc:.4f}")

plt.plot(k_vals, acc_scores, marker='o')
plt.xlabel('k (number of components)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k')
plt.axvline(best_k, color='r', linestyle='--')
plt.savefig(os.path.join(SAVE_DIR, 'acc_vs_k.png'))
plt.close()

# Train final model with best_k
print(f"Training final model with k={best_k}")
X_tr_final = omega_train[:best_k, :].T
X_te_final = omega_test[:best_k, :].T

model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=seed)
model.fit(X_tr_final, y_train_num)

preds = model.predict(X_te_final)
print("Final accuracy:", accuracy_score(y_test_num, preds))

# confusion matrix
cm = confusion_matrix(y_test_num, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues')
plt.xticks(rotation=45)
plt.savefig(os.path.join(SAVE_DIR, 'confusion.png'))
plt.close()

# Imposter test using Distance in Face Space (DIFS)
print("Running imposter detection test using Latent Space Distance...")
# Compute class centroids in the latent space
class_centroids = []
for c in range(len(valid_classes)):
    idx = np.where(y_train_num == c)[0]
    centroid = X_tr_final[idx].mean(axis=0)
    class_centroids.append(centroid)
class_centroids = np.array(class_centroids)

# Predict classes using the ANN
preds_test = model.predict(X_te_final)
X_imp_final = omega_imp[:best_k, :].T
preds_imp = model.predict(X_imp_final)

# Compute distances from test samples to their predicted class centroid
dists_test = np.array([np.linalg.norm(X_te_final[i] - class_centroids[preds_test[i]]) for i in range(len(X_te_final))])
# Set a dynamic threshold: e.g., 90th percentile of the known test set distances
threshold_dist = np.percentile(dists_test, 90)

print(f"Dynamic distance threshold set to {threshold_dist:.4f}")

# Compute distances for imposters
dists_imp = np.array([np.linalg.norm(X_imp_final[i] - class_centroids[preds_imp[i]]) for i in range(len(X_imp_final))])

rejected = sum(dists_imp > threshold_dist)
total = len(dists_imp)

print(f"Imposter rejection rate: {rejected}/{total} ({(rejected/total)*100:.1f}%)")

plt.figure(figsize=(8, 6))
plt.hist(dists_imp, bins=10, alpha=0.7, label='Imposters')
plt.hist(dists_test, bins=10, alpha=0.7, label='Enrolled Test')
plt.axvline(threshold_dist, color='r', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel('Distance to Predicted Class Centroid')
plt.ylabel('Count')
plt.title('Latent Space Distances: Enrolled vs Imposter')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, 'imposter_dist.png'))
plt.close()

print("Done. Results saved in", SAVE_DIR)
