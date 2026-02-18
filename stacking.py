import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
plt.rcParams['font.family'] = 'Times New Roman'

whole_data = pd.read_csv('C:/whole.csv')
filtered_features = pd.read_csv('C:/filtered_features.csv')

data = pd.concat([whole_data, filtered_features], axis=1)
fp_columns = [col for col in data.columns if col.startswith('FP_')]
X = data[['P (bar)', 'T (K)'] + fp_columns]
y = data['x_CO2'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

morgan_columns = fp_columns
other_columns = ['P (bar)', 'T (K)']

X_morgan = X[morgan_columns]

X_other = X[other_columns]

scaler_X = StandardScaler()
X_other_scaled = scaler_X.fit_transform(X_other)

X_scaled = pd.DataFrame(X_other_scaled, columns=other_columns)
X_scaled = pd.concat([X_scaled, X_morgan], axis=1)

X_train_scaled = X_scaled.iloc[X_train.index, :]
X_val_scaled = X_scaled.iloc[X_val.index, :]
X_test_scaled = X_scaled.iloc[X_test.index, :]

X_train_tensor = torch.FloatTensor(X_train_scaled.values)
X_val_tensor = torch.FloatTensor(X_val_scaled.values)
X_test_tensor = torch.FloatTensor(X_test_scaled.values)
y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor = torch.FloatTensor(y_val)
y_test_tensor = torch.FloatTensor(y_test)

hyperparameters = {
    'dropout_rate': 0.1,
    'learning_rate': 0.002895,
    'batch_size': 64,
    'weight_decay': 0.000001,
    'epochs': 100
}

class NN(nn.Module):
    def __init__(self, n_features, dropout_rate):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(n_features, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


model_nn = NN(n_features=X_train.shape[1], dropout_rate=hyperparameters['dropout_rate'])
optimizer = optim.Adam(model_nn.parameters(), lr=hyperparameters['learning_rate'],
                       weight_decay=hyperparameters['weight_decay'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
criterion = nn.MSELoss()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

train_losses = []
val_losses = []
for epoch in range(100):
    model_nn.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    scheduler.step()

    train_losses.append(epoch_train_loss / len(train_loader))

    model_nn.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model_nn(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}')


class NNBaseEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, model, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self._estimator_type = "regressor"

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        return predictions

    def get_params(self, deep=True):
        return {"model": self.model, "batch_size": self.batch_size}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

final_lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1695,
    max_depth= 20,
    num_leaves=45,
    random_state=42
    )
nn_model = NNBaseEstimator(model_nn)

stacking_model = StackingRegressor(
    estimators=[('nn', nn_model), ('lgbm', final_lgbm)],
    final_estimator=LinearRegression(),
    cv=10
)

stacking_model.fit(X_train_scaled.values, y_train)


y_train_stacking_pred = stacking_model.predict(X_train_scaled.values)
y_val_stacking_pred = stacking_model.predict(X_val_scaled.values)
y_test_stacking_pred = stacking_model.predict(X_test_scaled.values)


def print_metrics(model_name, y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - {dataset_name}: R² = {r2:.4f}, MSE = {mse:.6f}, MAE = {mae:.6f}")


print_metrics("Stacking", y_train, y_train_stacking_pred, "训练集")
print_metrics("Stacking", y_val, y_val_stacking_pred, "验证集")
print_metrics("Stacking", y_test, y_test_stacking_pred, "测试集")


train_errors = y_train - y_train_stacking_pred
test_errors = y_test - y_test_stacking_pred


plt.figure(figsize=(12, 8))
all_true = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_train_stacking_pred,  y_test_stacking_pred])
min_val = min(all_true.min(), all_pred.min()) - 0.1
max_val = max(all_true.max(), all_pred.max()) + 0.1

plt.scatter(y_train, y_train_stacking_pred, c='#1f77b4', alpha=0.6, label='Train')
plt.scatter(y_test,  y_test_stacking_pred, c='#ff7f0e', alpha=0.6, label='Test')
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.gca().set_aspect('equal')
plt.xlabel('True Values', fontsize=22)
plt.ylabel('Predicted Values', fontsize=22)

ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)
plt.legend(loc='best',fontsize=16)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
x_min, x_max = -0.25, 0.25
bins = np.linspace(x_min, x_max, 101)
train_counts, train_bins, _ = plt.hist(train_errors, bins=bins, alpha=0.6, color='steelblue',
                                       edgecolor='black', label=f'Train set')
test_counts, test_bins, _ = plt.hist(test_errors, bins=bins, alpha=0.6, color='coral',
                                     edgecolor='black', label=f'Test Set')
plt.xlim(x_min, x_max)

x = np.linspace(x_min, x_max, 1000)


train_mean = np.mean(train_errors)
train_std = np.std(train_errors)
train_norm = norm.pdf(x, train_mean, train_std)


bin_width = train_bins[1] - train_bins[0]
train_norm_scaled = train_norm * len(train_errors) * bin_width


plt.plot(x, train_norm_scaled, color='navy', linewidth=2, alpha=0.8,linestyle='--')
test_mean = np.mean(test_errors)
test_std = np.std(test_errors)
test_norm = norm.pdf(x, test_mean, test_std)

test_norm_scaled = test_norm * len(test_errors) * bin_width


plt.plot(x, test_norm_scaled, color='darkred', linewidth=2, alpha=0.8,linestyle='--')

plt.xlabel('Error of Stacking', fontsize=22)
plt.ylabel('Number of data points in each error range', fontsize=22)
plt.tick_params(axis='both', which='major',  labelsize=20)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 8),constrained_layout=True)
fixed_x_min = 0
fixed_x_max = 1.0
scatter1 = axes[0].scatter(y_train_stacking_pred, train_errors, alpha=0.6,
                          c=np.abs(train_errors), cmap='viridis', s=30)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=2)
axes[0].set_ylim(-0.5, 0.5)
axes[0].set_xlim(fixed_x_min, fixed_x_max)  # 添加这一行！
axes[0].set_xlabel('Train Predicted Values', fontsize=22)
axes[0].set_ylabel('Error', fontsize=22)


scatter2 = axes[1].scatter(y_test_stacking_pred, test_errors, alpha=0.6,
                          c=np.abs(test_errors), cmap='viridis', s=30)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=2)
axes[1].set_ylim(-0.5, 0.5)
axes[1].set_xlim(fixed_x_min, fixed_x_max)
axes[1].set_xlabel('Test Predicted Values', fontsize=22)

for ax in axes:
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)


cbar = plt.colorbar(scatter2, ax=axes[1], label='Absolute Error')
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Absolute Error', fontsize=22)
plt.show()
